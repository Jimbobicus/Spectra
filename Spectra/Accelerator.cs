using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using static Spectra.Accelerator.Side;
namespace Spectra;

unsafe struct TriList
{
    public uint GetTriCount() { return m_nTris; }

    public PreprocessedTri* GetTriList() { return (PreprocessedTri*)(Unsafe.Add<TriList>(Unsafe.AsPointer(ref this), 1)); }
    public uint m_nTris;
}


unsafe struct BVHNode
{
    public bool IsLeaf() { return (m_pNext & 4) != 0; }

    public ulong GetSplitAxis() { return m_pNext & 3u; }
    public byte* GetPrefetch() { return (byte*)m_pNext; }
    //public float* GetAABB() { return m_AABB; }
    public TriList* GetTriList() { return (TriList*)(m_pNext & ~7u); }
    public BVHNode* GetLeftChild() { return (BVHNode*)(m_pNext & ~7u); }
    public BVHNode* GetRightChild() { return GetLeftChild() + 1; }

    public void SetAABB(float* p)
    {
        for (int i = 0; i < 6; i++)
            m_AABB[i] = p[i];
    }

    public void CreateLeaf(TriList* pList)
    {
        m_pNext = (ulong)pList;
        m_pNext |= 4;
    }

    public void CreateInnerNode(BVHNode* pKids, uint nAxis)
    {
        m_pNext = (ulong)pKids;
        m_pNext |= (nAxis);
    }


    public fixed float m_AABB[6];
    ulong m_pNext; ///< Pointer to pair of child nodes (inner), or triangle list (leaf)
                   ///< low-order bits contain 2-bit split axis and 1 bit (is-leaf) flag 
}



unsafe struct Accelerator
{
    public BVHNode* pBVHRoot;
    public long nStackDepth;
    public long nRefCount;
    public PoolAllocator Memory;

    public enum Side
    {
        LEFT,
        RIGHT
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct PrimData
    {
        [FieldOffset(0)]
        public uint nTriangleID;
        [FieldOffset(4)]
        public fixed int nNextByAxis[3];    // Sorted linked-lists of prims, by axis.  stored as offsets to next

        /* union */
        [FieldOffset(16)]
        public float fLeftCost;         // Cost of placing this and previous prims on the left.
        [FieldOffset(16)]
        public float fCentroid;         // AABB centroid, during the initial sort-by-axis
        /* union */

        [FieldOffset(20)]
        public byte nSide;
        [FieldOffset(21)]
        fixed byte pad[3]; // pad to multiple of 4 bytes

        [FieldOffset(24)]
        public AxisAlignedBox aabb;

        public void Prefetch() => Sse.Prefetch0((char*)Unsafe.AsPointer(ref this));

        public PrimData* GetNextByAxis(int nAxis) => ApplyOffset(nNextByAxis[nAxis]);
        public void SetNextByAxis(int nAxis, PrimData* pNext) { nNextByAxis[nAxis] = GetOffset(pNext); }


        PrimData* ApplyOffset(int n) { return (PrimData*)((uint*)Unsafe.AsPointer(ref this) + n); }
        int GetOffset(PrimData* p) { return (int)(((uint*)p) - (uint*)Unsafe.AsPointer(ref this)); }
    }


    struct SplitInfo
    {
        public int nAxis;
        public float fSplitCost;
        public uint nLeftCount;        ///< Number of objects to the left
        public float fRightTotalCost;
        public float fLeftTotalCost;
        public AxisAlignedBox rightSideAABB;
    }

    /// Creates a degenerate AABB
    static AxisAlignedBox DegenerateBox()
    {
        return new AxisAlignedBox(new(float.MaxValue), new(float.MinValue));
    }

    /// Computes half the box's surface area
    static float Area(AxisAlignedBox box)
    {
        Vector3 vBBSize = box.Max - box.Min;
        return ((vBBSize.X * (vBBSize.Y + vBBSize.Z) + (vBBSize.Y * vBBSize.Z)));
    }

    static float* GetVert(Mesh* pMesh, uint i)
    {
        byte* pVerts = (byte*)pMesh->pVertexPositions;
        return (float*)(pVerts + i * pMesh->nVertexStride);
    }

    static uint GetIndex(Mesh* pMesh, uint tri, uint i)
    {
        byte* pIndices = (byte*)pMesh->pIndices;
        return *(uint*)(pIndices + tri * pMesh->nTriangleStride + i * sizeof(uint));
    }


    /// Chooses the SAH-optimal split on a particular axis, and fills in the rSplit structure if a better split is found
    static void SplitSelect(int nAxis,
                             float fInvRootArea,
                             float fIntersectCost,
                             PrimData* pFirstPrim,
                             uint nPrimitives,
                             ref SplitInfo rSplit)
    {

        // ------------------------------------------------------------------------------------------------------------
        // sweep left, and compute cost of left subtree when each object and its predecessors are put on the left side
        // ------------------------------------------------------------------------------------------------------------
        float fAccumulatedCost = 0;
        AxisAlignedBox leftBox = DegenerateBox();

        PrimData* pPrev = pFirstPrim;
        for (uint i = 0; i < nPrimitives; i++)
        {
            // merge in a new primitive
            PrimData* pNextPrim = pFirstPrim->GetNextByAxis(nAxis);
            pNextPrim->Prefetch();

            leftBox.Merge(pFirstPrim->aabb);
            fAccumulatedCost += fIntersectCost;

            // compute cost of putting this object (and everything to its left) in the left subtree
            float fLeftArea = Area(leftBox);
            pFirstPrim->fLeftCost = fLeftArea * fAccumulatedCost;

            // advance to next node, and reverse pointers on current node so we can sweep in reverse during next pass
            pFirstPrim->SetNextByAxis(nAxis, pPrev);
            pPrev = pFirstPrim;
            pFirstPrim = pNextPrim;
        }

        // ------------------------------------------------------------------------------------------------------------
        // sweep right to left, compute right-side costs and do final split-selection
        // ------------------------------------------------------------------------------------------------------------
        pFirstPrim = pPrev;

        float fTotalCost = fAccumulatedCost;

        // get AABB and cost for right-most primitive, and start sweep with its successor 
        AxisAlignedBox rightBox = pFirstPrim->aabb;
        fAccumulatedCost = fIntersectCost;

        pFirstPrim = pFirstPrim->GetNextByAxis(nAxis);

        for (uint i = 1; i < nPrimitives; i++)
        {
            PrimData* pNextPrim = pFirstPrim->GetNextByAxis(nAxis);
            pNextPrim->Prefetch();

            // compute true cost of split where this object and all subsequent ones are on the LEFT
            Vector3 vBoxSize = rightBox.Max - rightBox.Min;
            float fRightArea = Area(rightBox);
            float fSAHCost = 2.0f + (pFirstPrim->fLeftCost + fRightArea * fAccumulatedCost) * fInvRootArea;

            // save split if its better
            if (fSAHCost < rSplit.fSplitCost)
            {
                rSplit.nAxis = nAxis;
                rSplit.fSplitCost = fSAHCost;
                rSplit.nLeftCount = (uint)(nPrimitives - i);
                rSplit.rightSideAABB = rightBox;
                rSplit.fRightTotalCost = fAccumulatedCost;
                rSplit.fLeftTotalCost = fTotalCost - rSplit.fRightTotalCost;
            }

            // merge in this primitive
            rightBox.Merge(pFirstPrim->aabb);
            fAccumulatedCost += fIntersectCost;


            // advance to next node, and reverse pointers on current node so we can sweep in reverse during next pass
            pFirstPrim->SetNextByAxis(nAxis, pPrev);
            pPrev = pFirstPrim;
            pFirstPrim = pNextPrim;
        }

    }


    /// Partitions one of the axis-sorted lists into left and right halves.  Assumes that at least one prim goes in each sub-list
    static void UnsortedAxisPartition(int nAxis, PrimData* pPrims, uint nPrimitives, PrimData** pHeads)
    {
        PrimData** pTails = stackalloc PrimData*[2];

        // find the head in each of the lists
        byte nSide = pPrims->nSide;

        // first primitive is head of its side's list.  Iterate until the side changes (it must) 
        pHeads[nSide] = pPrims;
        PrimData* pPrev = null;
        while (true)// loop is infinite so that termination test can overlap prefetch
        {
            PrimData* pNext = pPrims->GetNextByAxis(nAxis);
            pNext->Prefetch();

            if (pPrims->nSide != nSide)
                break;

            pPrev = pPrims;
            nPrimitives--;
            pPrims = pNext;

        }

        pTails[nSide] = pPrev;

        // the current primitive is now the head (and tail) of the opposite side list. 
        nSide = (byte)(nSide ^ 1);
        pHeads[nSide] = pPrims;

        // keep splicing runs of like-sided primitives until we run out
        while (nPrimitives != 0)
        {
            // walk past run of like primitives
            //PrimData* pPrev;
            while (true)
            {
                PrimData* pNext = pPrims->GetNextByAxis(nAxis);
                pNext->Prefetch();
                if ((nPrimitives == 0) || pPrims->nSide != nSide)
                    break;

                pPrev = pPrims;
                nPrimitives--;
                pPrims = pNext;
            }

            // record the tail of the list
            pTails[nSide] = pPrev;

            // switch sides, and splice next node onto opposite side list
            nSide = (byte)(nSide ^ 1);
            pTails[nSide]->SetNextByAxis(nAxis, pPrims);
        }
    }


    /// Partitions the axis-sorted list on the split axis, and computes the left side AABB and cost
    static void SortedAxisPartition(PrimData** pObjectsByAxis,
                                     uint nPrimitives,
                                        ref SplitInfo split,
                                     PrimData** pOutputLists,
                                     ref AxisAlignedBox rLeftBox)
    {
        rLeftBox = DegenerateBox();

        // TODO: Could modify this to only one half (store the split node)

        // mark out the left half of the list, recompute the left-side AABB and cost
        uint nLeftCount = 0;
        PrimData* pFirst = pObjectsByAxis[split.nAxis];
        PrimData* pPrev = pFirst;
        pOutputLists[(int)LEFT] = pFirst;
        do
        {
            // prefetch next node in linked list
            PrimData* pNext = pFirst->GetNextByAxis(split.nAxis);
            pNext->Prefetch();

            // set the 'side' bit, and recompute left side AABB and total cost
            pFirst->nSide = (int)LEFT;
            rLeftBox.Merge(pFirst->aabb);

            pFirst = pNext;
            nLeftCount++;

        } while (nLeftCount < split.nLeftCount);

        // mark out the right half of the list
        nPrimitives -= nLeftCount;
        pOutputLists[(int)RIGHT] = pFirst;
        while (nPrimitives != 0)
        {
            pFirst->nSide = (int)RIGHT;
            pFirst = pFirst->GetNextByAxis(split.nAxis);
            nPrimitives--;
        }

    }


    static uint RecursiveBuild(Mesh* pMesh,
                                  ref PoolAllocator rMemory,
                                  BVHNode* pRoot,
                                     ref AxisAlignedBox rRootAABB,
                                  float fTotalCost,
                                  float fIntersectCost,
                                  PrimData** pObjectsByAxis,
                                  uint nPrimitives,
                                  ref float rSAH)
    {

        // select split
        SplitInfo split = new();
        split.fSplitCost = float.MaxValue;
        split.nAxis = 3;

        // compute inverse surface area of root AABB (off by 2, but thats ok...)
        float fInvRootArea = 1.0f / Area(rRootAABB);

        // sweep primitives and find the best split 
        for (int i = 0; i < 3; i++)
            SplitSelect(i, fInvRootArea, fIntersectCost, pObjectsByAxis[i], nPrimitives, ref split);


        pRoot->m_AABB[0] = rRootAABB.Min.X;
        pRoot->m_AABB[1] = rRootAABB.Min.Y;
        pRoot->m_AABB[2] = rRootAABB.Min.Z;
        pRoot->m_AABB[3] = rRootAABB.Max.X;
        pRoot->m_AABB[4] = rRootAABB.Max.Y;
        pRoot->m_AABB[5] = rRootAABB.Max.Z;


        // make a leaf if that's optimal
        if (split.fSplitCost >= fTotalCost)
        {
            long nAllocationSize = sizeof(TriList) + sizeof(PreprocessedTri) * nPrimitives;
            nAllocationSize = (nAllocationSize + 63) & (~63); // pad out to next cacheline
            byte* pLeafBytes = (byte*)rMemory.GetMore(nAllocationSize);
            TriList* pList = (TriList*)pLeafBytes;
            PreprocessedTri* pT = (PreprocessedTri*)(pList + 1);
            pList->m_nTris = (uint)nPrimitives;

            PrimData* pPrim = pObjectsByAxis[0];
            for (uint i = 0; i < nPrimitives; i++)
            {
                uint nTriIdx = pPrim->nTriangleID;
                float* pV0 = GetVert(pMesh, GetIndex(pMesh, nTriIdx, 0));
                float* pV1 = GetVert(pMesh, GetIndex(pMesh, nTriIdx, 1));
                float* pV2 = GetVert(pMesh, GetIndex(pMesh, nTriIdx, 2));
                Tracer.AssembleTri(pT + i, nTriIdx, pV0, pV1, pV2);
                pPrim = pPrim->GetNextByAxis(0);
            }

            pRoot->CreateLeaf(pList);

            rSAH = fTotalCost;
            return 1;
        }
        else
        {
            PrimData** pPartitionedByAxis = stackalloc PrimData*[3 * 2];

            // partition the axis-sorted object lists on the split axis.
            //  While partitioning, mark objects as 'left' or 'right' based on location relative to partition plane.
            //  also, recompute the left-side AAbb
            AxisAlignedBox leftBox = default;
            SortedAxisPartition(pObjectsByAxis, nPrimitives, ref split, &pPartitionedByAxis[split.nAxis * 2], ref leftBox);

            // Partition the axis-sorted lists on the other two axes
            int nA0 = (split.nAxis + 1) % 3;
            int nA1 = (split.nAxis + 2) % 3;
            UnsortedAxisPartition(nA0, pObjectsByAxis[nA0], nPrimitives, &pPartitionedByAxis[nA0 * 2]);
            UnsortedAxisPartition(nA1, pObjectsByAxis[nA1], nPrimitives, &pPartitionedByAxis[nA1 * 2]);


            // transpose the axis array so we can pass it to the recursive build calls
            PrimData** pAxisLists = stackalloc PrimData*[2 * 3];

            pAxisLists[0] = pPartitionedByAxis[0][0], pPartitionedByAxis[1][0], pPartitionedByAxis[2][0] },
                                           { pPartitionedByAxis[0][1], pPartitionedByAxis[1][1], pPartitionedByAxis[2][1] }
    }



    // Create an inner node, and do recursive tree build calls

    BVHNode* pKids = rMemory.GetSome<BVHNode>(2);
    BVHNode* pLeft = pKids;
    BVHNode* pRight = pKids + 1;
    pRoot->CreateInnerNode(pKids, split.nAxis);

    float fSAHLeft, fSAHRight;
    uint nLeftDepth = RecursiveBuild(pMesh, rMemory, pLeft, leftBox, split.fLeftTotalCost, fIntersectCost, pAxisLists[LEFT],
                                       split.nLeftCount, fSAHLeft);

    uint nRightDepth = RecursiveBuild(pMesh, rMemory, pRight, split.rightSideAABB, split.fRightTotalCost, fIntersectCost, pAxisLists[RIGHT],
                                        nPrimitives - split.nLeftCount, fSAHRight);

    float fLeftArea = Area(leftBox) * fInvRootArea;
    float fRightArea = Area(split.rightSideAABB) * fInvRootArea;

    rSAH = 2.0f + fLeftArea* fSAHLeft + fRightArea* fSAHRight;
            return 1 + Max(nLeftDepth, nRightDepth);
}
    }


    /*
        /// Helper function to sanity-check the sort routine
        static void ValidateSort( uint a, PrimData* pHead, uint n )
        {
            PrimData* pPrev = pHead;
            pHead = pHead->GetNextByAxis(a);
            n--;
            while( n )
            {
                assert( pPrev->fLeftCost <= pHead->fLeftCost );
                pPrev = pHead;
                pHead = pHead->GetNextByAxis(a);
                n--;
            }
        }
        */




    /// Sorts the per-axis primitive lists, and returns the head of a particular list
    static PrimData* MergeSortAxisList(uint nAxis, PrimData* pPrimData, uint nPrims)
{
    // pick the head of the merged list (left or right halves)
    if (nPrims <= 1)
        return pPrimData;

    PrimData* pLeftHalf = pPrimData;
    uint nLeft = nPrims / 2;
    PrimData* pRightHalf = pPrimData + nLeft;
    uint nRight = nPrims - nLeft;

    pLeftHalf = MergeSortAxisList(nAxis, pLeftHalf, nLeft);
    pRightHalf = MergeSortAxisList(nAxis, pRightHalf, nRight);

    PrimData* pSortedHead;
    if (pLeftHalf->fCentroid < pRightHalf->fCentroid)
    {
        pSortedHead = pLeftHalf;
        nLeft--;
        pLeftHalf = pLeftHalf->GetNextByAxis(nAxis);
    }
    else
    {
        pSortedHead = pRightHalf;
        nRight--;
        pRightHalf = pRightHalf->GetNextByAxis(nAxis);
    }

    // merge lists until we run out on one side or the other
    PrimData* pSortedTail = pSortedHead;
    while (nLeft && nRight)
    {
        if (pLeftHalf->fCentroid < pRightHalf->fCentroid)
        {
            PrimData* pFirst = pLeftHalf;
            pSortedTail->SetNextByAxis(nAxis, pLeftHalf);
            float fRight = pRightHalf->fCentroid;

            // find end of sorted sub-section
            while (1)
            {
                PrimData* pN = pLeftHalf->GetNextByAxis(nAxis);
                pN->Prefetch();

                if (!nLeft || pLeftHalf->fCentroid >= fRight)
                    break; // termination test is overlapped with next-node prefetch

                nLeft--;
                pSortedTail = pLeftHalf;
                pLeftHalf = pN;
            }
        }
        else
        {
            PrimData* pFirst = pRightHalf;
            pSortedTail->SetNextByAxis(nAxis, pRightHalf);
            float fLeft = pLeftHalf->fCentroid;

            // find end of sorted sub-section
            while (1)
            {
                PrimData* pN = pRightHalf->GetNextByAxis(nAxis);
                pN->Prefetch();

                if (!nRight || fLeft < pRightHalf->fCentroid)
                    break;  // termination test is overlapped with next-node prefetch

                nRight--;
                pSortedTail = pRightHalf;
                pRightHalf = pN;
            }
        }
    }

    // splice remaining sorted list onto end
    if (nLeft)
        pSortedTail->SetNextByAxis(nAxis, pLeftHalf);
    if (nRight)
        pSortedTail->SetNextByAxis(nAxis, pRightHalf);

    return pSortedHead;
}




void BuildBVH(Accelerator* pAccel, Mesh* pMesh, float fIntersectCost)
{
    // build PrimData structures
    uint nTotalPrims = pMesh->nTriangles;

    PrimData* pPrimData = (PrimData*)_INTERNAL::Malloc(sizeof(PrimData) * nTotalPrims, BRAY_MEM_ALIGNMENT);

    // fill in primtitive locations and compute AABBs
    AxisAlignedBox rootAABB = DegenerateBox();
    float fRootCost = fIntersectCost * nTotalPrims;

    PrimData* pCur = pPrimData;
    for (uint i = 0; i < nTotalPrims; i++)
    {
        pPrimData[i].nTriangleID = (uint)i;

        uint* pIndices = pMesh->pIndices + 3 * i;
        float* pV0 = GetVert(pMesh, GetIndex(pMesh, i, 0));
        float* pV1 = GetVert(pMesh, GetIndex(pMesh, i, 1));
        float* pV2 = GetVert(pMesh, GetIndex(pMesh, i, 2));
        pPrimData[i].aabb.Min.x = Min(pV0[0], Min(pV1[0], pV2[0]));
        pPrimData[i].aabb.Max.x = Max(pV0[0], Max(pV1[0], pV2[0]));
        pPrimData[i].aabb.Min.y = Min(pV0[1], Min(pV1[1], pV2[1]));
        pPrimData[i].aabb.Max.y = Max(pV0[1], Max(pV1[1], pV2[1]));
        pPrimData[i].aabb.Min.z = Min(pV0[2], Min(pV1[2], pV2[2]));
        pPrimData[i].aabb.Max.z = Max(pV0[2], Max(pV1[2], pV2[2]));

        rootAABB.Merge(pPrimData[i].aabb);
    }

    // sort by axis
    PrimData* pFirstByAxis[3];
    for (uint nAxis = 0; nAxis < 3; nAxis++)
    {
        // initialize per-axis lists
        for (uint i = 0; i < nTotalPrims; i++)
        {
            float* pBBMin = &pPrimData[i].aabb.Min.x;
            float* pBBMax = &pPrimData[i].aabb.Max.x;
            pPrimData[i].fCentroid = (pBBMin[nAxis] + pBBMax[nAxis]) * 0.5f;
            pPrimData[i].SetNextByAxis(nAxis, &pPrimData[i + 1]);
        }

        // sort the lists
        pFirstByAxis[nAxis] = MergeSortAxisList(nAxis, pPrimData, nTotalPrims);
    }


    // extra node next to root to force child pairs into same cacheline
    BVHNode* pRoot = pAccel->Memory.GetSome<BVHNode>(2);
    pAccel->pBVHRoot = pRoot;


    // do recursive build
    float fCost;
    uint nBVHDepth = RecursiveBuild(pMesh, pAccel->Memory, pRoot, rootAABB, fRootCost, fIntersectCost, pFirstByAxis, nTotalPrims, fCost);

    pAccel->nStackDepth = nBVHDepth;

    _INTERNAL::Free(pPrimData);
}
}

