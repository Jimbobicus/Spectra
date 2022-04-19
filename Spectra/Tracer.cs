using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

using static Spectra.Raytracer;
using static System.Runtime.Intrinsics.X86.Avx;
using static System.Runtime.Intrinsics.X86.Avx2;
using static System.Runtime.Intrinsics.X86.Fma;

namespace Spectra;

struct Ray
{
    public float ox;
    public float oy;
    public float oz;
    public float tmax;
    public float dx;
    public float dy;
    public float dz;
    public uint offset; ///< Byte offset of this ray from start of stream
}

unsafe struct PreprocessedTri
{
    public uint nID;
    public fixed float P0[3];
    public fixed float v10[3];
    public fixed float v02[3];
    public fixed float v10x02[3];
}

unsafe struct StackEntry
{
    public BVHNode* pNode;
    public long nRayPop;
    public long nGroups;
}


unsafe partial struct Tracer
{
    public AcceleratorHandle pAccelerator;
    public uint nMaxRays;

    public Ray* pRays;
    public BVHNode* pBVHRoot;
    public StackEntry* pTraversalStack;

    public uint nRays;
    public fixed ushort pOctantCounts[8];
    public byte* pRayOctants;

    public static void AssembleTri(PreprocessedTri* pTri, uint nID, float* P0, float* P1, float* P2)
    {
        pTri->nID = nID;
        pTri->P0[0] = P0[0];
        pTri->P0[1] = P0[1];
        pTri->P0[2] = P0[2];
        pTri->v10[0] = P1[0] - P0[0];
        pTri->v10[1] = P1[1] - P0[1];
        pTri->v10[2] = P1[2] - P0[2];
        pTri->v02[0] = P0[0] - P2[0];
        pTri->v02[1] = P0[1] - P2[1];
        pTri->v02[2] = P0[2] - P2[2];
        pTri->v10x02[0] = pTri->v10[1] * pTri->v02[2] - pTri->v10[2] * pTri->v02[1];
        pTri->v10x02[1] = pTri->v10[2] * pTri->v02[0] - pTri->v10[0] * pTri->v02[2];
        pTri->v10x02[2] = pTri->v10[0] * pTri->v02[1] - pTri->v10[1] * pTri->v02[0];
    }

    public static void DoTrace(ref Tracer pTracer, RayHitInfo* pHitInfoOut)
    {
        RayPacket* packs = stackalloc RayPacket[MAX_TRACER_SIZE / 8 + 8]; // up to one additional packet per octant

        StackFrame frame = new();
        frame.pRays = pTracer.pRays;
        frame.nRays = pTracer.nRays;
        frame.pAllPackets = packs;
        frame.pHitInfo = pHitInfoOut;
        frame.pBVH = pTracer.pBVHRoot;

        for (int i = 0; i < frame.nRays; i++)
            pHitInfoOut[i].nPrimID = RayHitInfo.NULL_PRIMID;

        uint* pPacksByOctant = stackalloc uint[8];
        BuildPacketsByOctant(packs, ref pTracer, pPacksByOctant);

        for (uint i = 0; i < 8; i++)
        {
            uint n = pPacksByOctant[i];
            if (n != 0)
            {
                frame.nPackets = n;
                AdaptiveTrace(pTracer.pTraversalStack, ref frame, i);
            }

            frame.pAllPackets += n;
        }
    }

    /// Intersector state for a packet is consolidated in a local struct so that hit info is not
    ///  constantly written back and causing cache pollution
    [StructLayout(LayoutKind.Sequential, Pack = 32)]
    unsafe struct PacketISectCache
    {
        //public Span<Pointer<float>> D => MemoryMarshal.CreateSpan(ref _D, 3);
        public fixed float D[24];
        public fixed float hA[8];
        public fixed float hB[8];
        public fixed uint ID[8];
        public ulong mask;
    }

    static void PrepIntersectCache(PacketISectCache* pInfo, RayPacket* pPacket, Ray* pRays)
    {
        pInfo->mask = 0;
        ReadDirs((Vector256<float>*)pInfo->D, pPacket, pRays);
    }

    static void WritebackIntersectCache(PacketISectCache* pInfo,
                                         RayPacket* pPacket,
                                         StackFrame* pFrame)
    {

        float* pT = (float*)&pPacket->TMax;
        RayHitInfo* pHitInfo = pFrame->pHitInfo;
        Ray* pRays = pFrame->pRays;

        // now write results back out
        ulong nHit = pInfo->mask;
        ulong i = Bmi1.X64.TrailingZeroCount(nHit);
        while (i < 32)
        {
            ulong id = (ulong)(pPacket->RayOffsets[i] / sizeof(Ray));
            float t = pT[i];
            pRays[id].tmax = t;
            pHitInfo[id].t = t;
            pHitInfo[id].u = pInfo->hA[i];
            pHitInfo[id].v = pInfo->hB[i];
            pHitInfo[id].nPrimID = pInfo->ID[i];

            nHit = Bmi1.X64.ResetLowestSetBit(nHit); // sets lowest bit t0 zero
            i = Bmi1.X64.TrailingZeroCount(nHit);
        }

    }



    static void TriISectPacket_ISectCache(TriList* pList,
                                           RayPacket* pPacket,
                                           PacketISectCache* pPrepped)
    {
        Vector256<float> TMax = pPacket->TMax;

        uint nTris = pList->GetTriCount();
        for (uint i = 0; i < nTris; i++)
        {
            PreprocessedTri* pTri = pList->GetTriList() + i;
            float* P0 = pTri->P0;
            float* v10 = pTri->v10;
            float* v02 = pTri->v02;
            float* v10x02 = pTri->v10x02;

            Sse.Prefetch0((char*)(pTri + 1));


            Vector256<float> vDx = LoadAlignedVector256((float*)&pPrepped->D[0 * 8]);
            Vector256<float> vDy = LoadAlignedVector256((float*)&pPrepped->D[1 * 8]);
            Vector256<float> vDz = LoadAlignedVector256((float*)&pPrepped->D[2 * 8]);
            Vector256<float> vOx = LoadAlignedVector256((float*)&pPacket->Ox);
            Vector256<float> vOy = LoadAlignedVector256((float*)&pPacket->Oy);
            Vector256<float> vOz = LoadAlignedVector256((float*)&pPacket->Oz);

            Vector256<float>[] v0A = new[] {
                Subtract( BroadcastScalarToVector256( &P0[0] ), vOx ),
                Subtract( BroadcastScalarToVector256( &P0[1] ), vOy ),
                Subtract( BroadcastScalarToVector256( &P0[2] ), vOz ),
            };


            //V = ((p1 - p0)x(p0 -p2)).d
            //Va = ((p1 - p0)x(p0 -p2)).(p0 -a)
            //V1 = ((p0 - a)×d).(p0 -p2)
            //V2 = ((p0 - a)×d).(p1 -p0)

            Vector256<float> v10x02_x = BroadcastScalarToVector256(&v10x02[0]);
            Vector256<float> v10x02_y = BroadcastScalarToVector256(&v10x02[1]);
            Vector256<float> v10x02_z = BroadcastScalarToVector256(&v10x02[2]);

            Vector256<float> T = MultiplyAdd(v10x02_z, v0A[2],
                                          MultiplyAdd(v10x02_y, v0A[1],
                                          Multiply(v10x02_x, v0A[0])));

            Vector256<float> V = MultiplyAdd(v10x02_z, vDz,
                            MultiplyAdd(v10x02_y, vDy,
                                                Multiply(v10x02_x, vDx)));
            V = RCPNR(V);

            Vector256<float> v02_x = BroadcastScalarToVector256(&v02[0]);
            Vector256<float> v02_y = BroadcastScalarToVector256(&v02[1]);
            Vector256<float> v02_z = BroadcastScalarToVector256(&v02[2]);

            Vector256<float>[] v0AxD = new[] {
                MultiplySubtract( v0A[1], vDz, Multiply( v0A[2], vDy ) ),
                MultiplySubtract( v0A[2], vDx, Multiply( v0A[0], vDz ) ),
                MultiplySubtract( v0A[0], vDy, Multiply( v0A[1], vDx ) )
            };
            Vector256<float> A = MultiplyAdd(v0AxD[2], v02_z,
                                         MultiplyAdd(v0AxD[1], v02_y,
                                         Multiply(v0AxD[0], v02_x)));

            Vector256<float> v10_x = BroadcastScalarToVector256(&v10[0]);
            Vector256<float> v10_y = BroadcastScalarToVector256(&v10[1]);
            Vector256<float> v10_z = BroadcastScalarToVector256(&v10[2]);
            Vector256<float> B = MultiplyAdd(v0AxD[2], v10_z,
                                         MultiplyAdd(v0AxD[1], v10_y,
                                         Multiply(v0AxD[0], v10_x)));


            A = Multiply(A, V);
            B = Multiply(B, V);
            T = Multiply(T, V);

            Vector256<float> front = And(CompareGreaterThan(T, Vector256<float>.Zero),
                                          CompareGreaterThan(TMax, T));
            Vector256<float> inside = And(CompareGreaterThanOrEqual(A, Vector256<float>.Zero),
                                                  CompareGreaterThanOrEqual(B, Vector256<float>.Zero));

            inside = And(inside, CompareLessThanOrEqual(Add(A, B), Vector256.Create(1.0f)));

            Vector256<float> hit = And(inside, front);

            uint nHit = (uint)MoveMask(hit);
            if (nHit != 0)
            {
                TMax = BlendVariable(TMax, T, hit);
                StoreAligned((float*)&pPacket->TMax, TMax);
                Vector256<float> stored_a = LoadAlignedVector256(pPrepped->hA);
                Vector256<float> stored_b = LoadAlignedVector256(pPrepped->hB);
                Vector256<float> stored_id = LoadAlignedVector256((float*)pPrepped->ID);
                Vector256<float> ID = BroadcastScalarToVector256((float*)&pTri->nID);
                stored_a = BlendVariable(stored_a, A, hit);
                stored_b = BlendVariable(stored_b, B, hit);
                stored_id = BlendVariable(stored_id, ID, hit);
                StoreAligned(pPrepped->hA, stored_a);
                StoreAligned(pPrepped->hB, stored_b);
                StoreAligned((float*)pPrepped->ID, stored_id);
                pPrepped->mask |= nHit;
            }
        }

    }



    static void TriISectPacket_Preproc_List(TriList* pList, RayPacket* pPacket, StackFrame* pFrame)
    {
        Vector256<float> hA = Vector256<float>.Zero;
        Vector256<float> hB = Vector256<float>.Zero;
        Vector256<float> h = Vector256<float>.Zero;
        Vector256<float> ID = Vector256<float>.Zero;
        Vector256<float> TMax = pPacket->TMax;

        Vector256<float>* D = stackalloc Vector256<float>[3];
        ReadDirs(D, pPacket, pFrame->pRays);


        long nTris = pList->GetTriCount();
        for (int i = 0; i < nTris; i++)
        {
            PreprocessedTri* pTri = pList->GetTriList() + i;
            float* P0 = pTri->P0;
            float* v10 = pTri->v10;
            float* v02 = pTri->v02;
            float* v10x02 = pTri->v10x02;

            Sse.Prefetch0((char*)(pTri + 1));

            Vector256<float> vDx = D[0];
            Vector256<float> vDy = D[1];
            Vector256<float> vDz = D[2];
            Vector256<float>[] v0A = new[] {
                Subtract( BroadcastScalarToVector256( &P0[0] ), pPacket->Ox ),
                Subtract( BroadcastScalarToVector256( &P0[1] ), pPacket->Oy ),
                Subtract( BroadcastScalarToVector256( &P0[2] ), pPacket->Oz ),
            };
            Vector256<float>[] v0AxD = new[] {
                MultiplySubtract( v0A[1], vDz, Multiply( v0A[2], vDy ) ),
                MultiplySubtract( v0A[2], vDx, Multiply( v0A[0], vDz ) ),
                MultiplySubtract( v0A[0], vDy, Multiply( v0A[1], vDx ) )
            };

            //V = ((p1 - p0)x(p0 -p2)).d
            //Va = ((p1 - p0)x(p0 -p2)).(p0 -a)
            //V1 = ((p0 - a)×d).(p0 -p2)
            //V2 = ((p0 - a)×d).(p1 -p0)

            Vector256<float> v10x02_x = BroadcastScalarToVector256(&v10x02[0]);
            Vector256<float> v10x02_y = BroadcastScalarToVector256(&v10x02[1]);
            Vector256<float> v10x02_z = BroadcastScalarToVector256(&v10x02[2]);

            Vector256<float> T = MultiplyAdd(v10x02_z, v0A[2],
                                          MultiplyAdd(v10x02_y, v0A[1],
                                          Multiply(v10x02_x, v0A[0])));

            Vector256<float> V = MultiplyAdd(v10x02_z, vDz,
                            MultiplyAdd(v10x02_y, vDy,
                                                Multiply(v10x02_x, vDx)));
            V = RCPNR(V);

            Vector256<float> v02_x = BroadcastScalarToVector256(&v02[0]);
            Vector256<float> v02_y = BroadcastScalarToVector256(&v02[1]);
            Vector256<float> v02_z = BroadcastScalarToVector256(&v02[2]);

            Vector256<float> A = MultiplyAdd(v0AxD[2], v02_z,
                                         MultiplyAdd(v0AxD[1], v02_y,
                                         Multiply(v0AxD[0], v02_x)));

            Vector256<float> v10_x = BroadcastScalarToVector256(&v10[0]);
            Vector256<float> v10_y = BroadcastScalarToVector256(&v10[1]);
            Vector256<float> v10_z = BroadcastScalarToVector256(&v10[2]);
            Vector256<float> B = MultiplyAdd(v0AxD[2], v10_z,
                                         MultiplyAdd(v0AxD[1], v10_y,
                                         Multiply(v0AxD[0], v10_x)));


            A = Multiply(A, V);
            B = Multiply(B, V);
            T = Multiply(T, V);

            Vector256<float> front = And(CompareGreaterThan(T, Vector256<float>.Zero),
                                          CompareGreaterThan(TMax, T));
            Vector256<float> inside = And(CompareGreaterThanOrEqual(A, Vector256<float>.Zero),
                                          CompareGreaterThanOrEqual(B, Vector256<float>.Zero));

            inside = And(inside, CompareLessThanOrEqual(Add(A, B), Vector256.Create(1.0f)));

            Vector256<float> hit = And(inside, front);

            hA = BlendVariable(hA, A, hit);
            hB = BlendVariable(hB, B, hit);
            TMax = BlendVariable(TMax, T, hit);
            h = Or(h, hit);
            ID = BlendVariable(ID, BroadcastScalarToVector256((float*)&pTri->nID), hit);
        }

        // now write results back out
        ulong nHit = (uint)MoveMask(h);
        if (nHit != 0)
        {
            // merge T values back into packet
            StoreAligned((float*)&pPacket->TMax, TMax);

            // scatter out results
            Vector256<float> U = hA;
            Vector256<float> V = hB;

            float* pU = stackalloc float[8];
            float* pV = stackalloc float[8];
            float* pT = stackalloc float[8];
            uint* pID = stackalloc uint[8];
            StoreAligned((float*)pU, U);
            StoreAligned((float*)pV, V);
            StoreAligned((float*)pT, TMax);
            StoreAligned((float*)pID, ID);

            RayHitInfo* pHitInfo = pFrame->pHitInfo;
            Ray* pRays = pFrame->pRays;

            ulong i = Bmi1.X64.TrailingZeroCount(nHit);
            do
            {
                long id = pPacket->RayOffsets[i] / sizeof(Ray);

                pRays[id].tmax = pT[i];

                pHitInfo[id].t = pT[i];
                pHitInfo[id].u = pU[i];
                pHitInfo[id].v = pV[i];
                pHitInfo[id].nPrimID = pID[i];

                nHit = Bmi1.X64.ResetLowestSetBit(nHit); // sets lowest bit t0 zero
                i = Bmi1.X64.TrailingZeroCount(nHit);
            } while (i < 32);
        }
    }




    //template<int OCTANT>
    static void PacketTrace_Octant(ref RayPacket pack, void* pStackBottom, BVHNode* pRoot, ref StackFrame frame, int OCTANT)
    {

        BVHNode** pStack = (BVHNode**)pStackBottom;
        *(pStack++) = pRoot;

        Sse.Prefetch0((char*)pRoot);
        PacketISectCache ISect;
        fixed (RayPacket* pPack = &pack)
            PrepIntersectCache(&ISect, pPack, frame.pRays);


        Vector256<float>[] OD = new[] {
            Multiply( pack.DInvx, pack.Ox ),
            Multiply( pack.DInvy, pack.Oy ),
            Multiply( pack.DInvz, pack.Oz ),
        };


        while (pStack != pStackBottom)
        {
            BVHNode* pN = *(--pStack);
            if (pN->IsLeaf())
            {
                // visit leaf
                TriList* pList = pN->GetTriList();
                //TriISectPacket_Preproc_List(pList, &pack, &frame );
                fixed (RayPacket* pPack = &pack)
                    TriISectPacket_ISectCache(pList, pPack, &ISect);
            }
            else
            {
                BVHNode* pNLeft = pN->GetLeftChild();
                BVHNode* pNRight = pN->GetRightChild();
                float* pLeft = (float*)pNLeft->m_AABB;
                float* pRight = (float*)pNRight->m_AABB;

                // fetch AABB.  each of the min/max vectors has the same data replicated into low and high lanes
                // bbmin(xyz), bbmax(xyz)
                BVHNode* pL = pN->GetLeftChild();
                BVHNode* pR = pN->GetRightChild();
                Vector256<float> min0 = BroadcastVector128ToVector256((pLeft));
                Vector256<float> min1 = BroadcastVector128ToVector256((pRight));
                Vector256<float> max0 = BroadcastVector128ToVector256((pLeft + 3));
                Vector256<float> max1 = BroadcastVector128ToVector256((pRight + 3));

                // swap planes based on octant
                Vector256<float> bbmin0 = Blend(min0, max0, (byte)(OCTANT | OCTANT << 4));
                Vector256<float> bbmax0 = Blend(max0, min0, (byte)(OCTANT | OCTANT << 4));
                Vector256<float> bbmin1 = Blend(min1, max1, (byte)(OCTANT | OCTANT << 4));
                Vector256<float> bbmax1 = Blend(max1, min1, (byte)(OCTANT | OCTANT << 4));

                // prefetch left/right children since we'll probably need them
                Sse.Prefetch0((char*)pNLeft->GetPrefetch());
                Sse.Prefetch0((char*)pNRight->GetPrefetch());

                // axis tests
                Vector256<float> D = LoadAlignedVector256((float*)Unsafe.AsPointer(ref pack.DInvx));
                Vector256<float> O = LoadAlignedVector256((float*)Unsafe.AsPointer(ref OD[0]));
                Vector256<float> Bmin0 = Permute(bbmin0, 0x00);
                Vector256<float> Bmax0 = Permute(bbmax0, 0x00);
                Vector256<float> Bmin1 = Permute(bbmin1, 0x00);
                Vector256<float> Bmax1 = Permute(bbmax1, 0x00);
                Vector256<float> tmin0 = MultiplySubtract(Bmin0, D, O);
                Vector256<float> tmax0 = MultiplySubtract(Bmax0, D, O);
                Vector256<float> tmin1 = MultiplySubtract(Bmin1, D, O);
                Vector256<float> tmax1 = MultiplySubtract(Bmax1, D, O);

                D = LoadAlignedVector256((float*)Unsafe.AsPointer(ref pack.DInvy));
                O = LoadAlignedVector256((float*)Unsafe.AsPointer(ref OD[1]));
                Bmin0 = Permute(bbmin0, 0x55); // 0101
                Bmax0 = Permute(bbmax0, 0x55);
                Bmin1 = Permute(bbmin1, 0x55);
                Bmax1 = Permute(bbmax1, 0x55);
                Vector256<float> t0 = MultiplySubtract(Bmin0, D, O);
                Vector256<float> t1 = MultiplySubtract(Bmax0, D, O);
                Vector256<float> t2 = MultiplySubtract(Bmin1, D, O);
                Vector256<float> t3 = MultiplySubtract(Bmax1, D, O);
                tmin0 = Max(tmin0, t0);
                tmax0 = Min(tmax0, t1);
                tmin1 = Max(tmin1, t2);
                tmax1 = Min(tmax1, t3);

                D = LoadAlignedVector256((float*)Unsafe.AsPointer(ref pack.DInvz));
                O = LoadAlignedVector256((float*)Unsafe.AsPointer(ref OD[2]));
                Bmin0 = Permute(bbmin0, 0xAA); // 1010
                Bmax0 = Permute(bbmax0, 0xAA);
                Bmin1 = Permute(bbmin1, 0xAA);
                Bmax1 = Permute(bbmax1, 0xAA);
                t0 = MultiplySubtract(Bmin0, D, O);
                t1 = MultiplySubtract(Bmax0, D, O);
                t2 = MultiplySubtract(Bmin1, D, O);
                t3 = MultiplySubtract(Bmax1, D, O);
                tmin0 = Max(tmin0, t0);
                tmax0 = Min(tmax0, t1);
                tmin1 = Max(tmin1, t2);
                tmax1 = Min(tmax1, t3);

                Vector256<float> limL = Min(tmax0, pack.TMax);
                Vector256<float> limR = Min(tmax1, pack.TMax);

                // using sign-bit trick for tmax >= 0
                t0 = CompareLessThanOrEqual(tmin0, limL);
                t1 = CompareLessThanOrEqual(tmin1, limR);
                Vector256<float> hitL = AndNot(tmax0, t0);
                Vector256<float> hitR = AndNot(tmax1, t1);

                long maskhitL = MoveMask(hitL);
                long maskhitR = MoveMask(hitR);
                long maskhitB = maskhitL & maskhitR;
                if (maskhitB != 0)
                {
                    Vector256<float> LFirst = CompareLessThan(tmin0, tmin1);
                    long lf = MoveMask(LFirst) & maskhitB;
                    long rf = ~lf & maskhitB;

                    if (Popcnt.X64.PopCount((ulong)lf) > Popcnt.X64.PopCount((ulong)rf))
                    {
                        pStack[0] = pNRight;
                        pStack[1] = pNLeft;
                        pStack += 2;
                    }
                    else
                    {
                        pStack[0] = pNLeft;
                        pStack[1] = pNRight;
                        pStack += 2;
                    }
                }
                else
                {
                    if (maskhitL != 0)
                    {
                        *(pStack++) = pNLeft;
                    }
                    if (maskhitR != 0)
                    {
                        *(pStack++) = pNRight;
                    }
                }
            }
        }

        fixed (RayPacket* pPack = &pack)
        fixed (StackFrame* pFrame = &frame)
            WritebackIntersectCache(&ISect, pPack, pFrame);
    }

    //     delegate void PTRACE_OCTANT(ref RayPacket pack, void* pStackBottom, BVHNode* pRoot, ref StackFrame frame);

    //     //typedef void (* PTRACE_OCTANT) (RayPacket & pack, void* pStackBottom, BVHNode * pRoot, StackFrame & frame);
    // static readonly PTRACE_OCTANT[] OctantDispatch = {
    //         new Action<
    //         // &PacketTrace_Octant<0>,
    //         // &PacketTrace_Octant<1>,
    //         // &PacketTrace_Octant<2>,
    //         // &PacketTrace_Octant<3>,
    //         // &PacketTrace_Octant<4>,
    //         // &PacketTrace_Octant<5>,
    //         // &PacketTrace_Octant<6>,
    //         // &PacketTrace_Octant<7>,
    //     };

    static void PacketTrace_Octant(ref RayPacket pack, uint octant, void* pStackBottom, BVHNode* pRoot, ref StackFrame frame)
    {
        PacketTrace_Octant(ref pack, pStackBottom, pRoot, ref frame, (int)octant);
        //OctantDispatch[octant](ref pack, pStackBottom, pRoot, ref frame);
    }


    /*
    static void PacketTrace_Octant( RayPacket& pack, uint octant, void* pStackBottom, BVHNode* pRoot, StackFrame& frame )
    {
        BVHNode** pStack = (BVHNode**) pStackBottom;
        *(pStack++) = pRoot;
        _mm_prefetch((char*)pRoot, _MM_HINT_T0 );
        PacketISectCache ISect;
        PrepIntersectCache( &ISect, &pack, frame.pRays );
        // positive sign:  first=0, last=3
        // negative sign:  first=3, last=0
        uint xfirst = (octant&1) ? 3 : 0;
        uint yfirst = (octant&2) ? 3 : 0;
        uint zfirst = (octant&4) ? 3 : 0;
        Vector256<float> OD[] = { 
            Multiply( pack.DInvx, pack.Ox ),
            Multiply( pack.DInvy, pack.Oy ),
            Multiply( pack.DInvz, pack.Oz ),
        };

            // pick min/max if octant sign is zero, reverse otherwise
        __m128i vOctant = _mm_set_epi32( 0, octant&4 ? 0xffffffff : 0, 
                                            octant&2 ? 0xffffffff : 0,
                                            octant&1 ? 0xffffffff : 0 );
        Vector256<float> octant_select = _mm256_castsi256_ps(_mm256_broadcastsi128_si256(vOctant));
        while( pStack != pStackBottom )
        {
            BVHNode* pN = *(--pStack);
            if( pN->IsLeaf() )
            {
                // visit leaf
                TriList* pList = pN->GetTriList();
                //TriISectPacket_Preproc_List(pList, &pack, &frame );
                TriISectPacket_ISectCache( pList, &pack, &ISect );
            }
            else
            {
                BVHNode* pNLeft  = pN->GetLeftChild();
                BVHNode* pNRight = pN->GetRightChild();
                float* pLeft  = (float*)pNLeft->GetAABB();
                float* pRight = (float*)pNRight->GetAABB();


                // fetch AABB
                // bbmin(xyz), bbmax(xyz)
                BVHNode* pL = pN->GetLeftChild();
                BVHNode* pR = pN->GetRightChild();
                Vector256<float> min0 = _mm256_broadcast_ps( (__m128*)(pLeft)   );
                Vector256<float> min1 = _mm256_broadcast_ps( (__m128*)(pRight)   );
                Vector256<float> max0 = _mm256_broadcast_ps( (__m128*)(pLeft+3) );
                Vector256<float> max1 = _mm256_broadcast_ps( (__m128*)(pRight+3) );
                // swap planes based on octant
                Vector256<float> bbmin0 = BlendVariable(min0, max0, octant_select );
                Vector256<float> bbmax0 = BlendVariable(max0, min0, octant_select );
                Vector256<float> bbmin1 = BlendVariable(min1, max1, octant_select );
                Vector256<float> bbmax1 = BlendVariable(max1, min1, octant_select );

                _mm_prefetch( (char*) pNLeft->GetPrefetch(),  _MM_HINT_T0 );
                _mm_prefetch( (char*) pNRight->GetPrefetch(), _MM_HINT_T0 );
                // axis tests
                Vector256<float> D = LoadAlignedVector256( (float*)&pack.DInvx);
                Vector256<float> O = LoadAlignedVector256( (float*) (OD+0) );
                Vector256<float> Bmin0 = Permute(bbmin0, 0x00);
                Vector256<float> Bmax0 = Permute(bbmax0, 0x00);
                Vector256<float> Bmin1 = Permute(bbmin1, 0x00);
                Vector256<float> Bmax1 = Permute(bbmax1, 0x00);
                Vector256<float> tmin0 = MultiplySubtract( Bmin0, D, O );
                Vector256<float> tmax0 = MultiplySubtract( Bmax0, D, O );
                Vector256<float> tmin1 = MultiplySubtract( Bmin1, D, O );
                Vector256<float> tmax1 = MultiplySubtract( Bmax1, D, O );
                D = LoadAlignedVector256( (float*)&pack.DInvy);
                O = LoadAlignedVector256( (float*) (OD+1) );
                Bmin0 = Permute(bbmin0, 0x55); // 0101
                Bmax0 = Permute(bbmax0, 0x55);
                Bmin1 = Permute(bbmin1, 0x55);
                Bmax1 = Permute(bbmax1, 0x55);
                Vector256<float> t0 = MultiplySubtract( Bmin0, D, O );
                Vector256<float> t1 = MultiplySubtract( Bmax0, D, O );
                Vector256<float> t2 = MultiplySubtract( Bmin1, D, O );
                Vector256<float> t3 = MultiplySubtract( Bmax1, D, O );
                tmin0 = Max( tmin0, t0 );
                tmax0 = Min( tmax0, t1 );
                tmin1 = Max( tmin1, t2 );
                tmax1 = Min( tmax1, t3 );
                D = LoadAlignedVector256( (float*)&pack.DInvz);
                O = LoadAlignedVector256( (float*) (OD+2) );               
                Bmin0 = Permute(bbmin0, 0xAA); // 1010
                Bmax0 = Permute(bbmax0, 0xAA);
                Bmin1 = Permute(bbmin1, 0xAA);
                Bmax1 = Permute(bbmax1, 0xAA);
                t0 = MultiplySubtract( Bmin0, D, O );
                t1 = MultiplySubtract( Bmax0, D, O );
                t2 = MultiplySubtract( Bmin1, D, O );
                t3 = MultiplySubtract( Bmax1, D, O );
                tmin0 = Max( tmin0, t0  );
                tmax0 = Min( tmax0, t1  );
                tmin1 = Max( tmin1, t2  );
                tmax1 = Min( tmax1, t3  );
                Vector256<float> limL  = Min( tmax0, pack.TMax );
                Vector256<float> limR  = Min( tmax1, pack.TMax );
                // using sign-bit trick for tmax >= 0
                t0 =  _mm256_cmp_ps( tmin0, limL, _CMP_LE_OQ );
                t1 =  _mm256_cmp_ps( tmin1, limR, _CMP_LE_OQ );
                Vector256<float> hitL = AndNot( tmax0, t0 );
                Vector256<float> hitR = AndNot( tmax1, t1 );
                size_t maskhitL   = MoveMask(hitL);
                size_t maskhitR   = MoveMask(hitR);
                size_t maskhitB = maskhitL & maskhitR;
                if( maskhitB )
                {
                    Vector256<float> LFirst = _mm256_cmp_ps( tmin0, tmin1, _CMP_LT_OQ );
                    size_t lf = MoveMask(LFirst) & maskhitB;
                    size_t rf = ~lf & maskhitB;
                    if( _mm_popcnt_u64( lf ) > _mm_popcnt_u64( rf ) )
                    {
                        pStack[0] = pNRight;
                        pStack[1] = pNLeft;
                        pStack += 2;
                    }
                    else
                    {
                        pStack[0] = pNLeft;
                        pStack[1] = pNRight;
                        pStack += 2;
                    }
                }
                else
                {
                    if( maskhitL )
                    {
                        *(pStack++) = pNLeft;
                    }
                    if( maskhitR )
                    {
                        *(pStack++) = pNRight;
                    }
                }
            }
        }
        WritebackIntersectCache( &ISect, &pack, &frame );
    }
    */

    struct Stack_AdaptiveTrace
    {
        public BVHNode* pN;
        public long nGroups;
        public long nRayPop;
    }

    static void AdaptiveTrace(void* pStackMem, ref StackFrame frame, uint nRayOctant)
    {
        BVHNode* pBVH = frame.pBVH;
        RayPacket* pPackets = frame.pAllPackets;
        uint nPackets = (uint)frame.nPackets;

        for (int i = 0; i < nPackets; i++)
            frame.pActivePackets[i] = frame.pAllPackets + i;

        Stack_AdaptiveTrace* pStackBottom = (Stack_AdaptiveTrace*)pStackMem;
        Stack_AdaptiveTrace* pStack = pStackBottom;
        pStack->nGroups = nPackets;
        pStack->pN = pBVH;
        pStack->nRayPop = 8 * nPackets;
        ++pStack;

        uint xfirst = (nRayOctant & 1) != 0 ? 3u : 0;
        uint yfirst = (nRayOctant & 2) != 0 ? 3u : 0;
        uint zfirst = (nRayOctant & 4) != 0 ? 3u : 0;

        while (pStack != pStackBottom)
        {
            Stack_AdaptiveTrace* pS = (--pStack);
            BVHNode* pN = pS->pN;
            uint nGroups = (uint)pS->nGroups;

            // pre-swizzle node AABB based on ray direction signs
            float* pNodeAABB = pN->m_AABB;
            frame.pAABB[0] = pNodeAABB[0 + xfirst];
            frame.pAABB[1] = pNodeAABB[1 + yfirst];
            frame.pAABB[2] = pNodeAABB[2 + zfirst];
            frame.pAABB[3] = pNodeAABB[0 + (xfirst ^ 3)];
            frame.pAABB[4] = pNodeAABB[1 + (yfirst ^ 3)];
            frame.pAABB[5] = pNodeAABB[2 + (zfirst ^ 3)];

            // intersection test with node, store hit masks
            long nHitPopulation;
            fixed (StackFrame* pFrame = &frame)
                nHitPopulation = GroupTest2X(pFrame, nGroups);

            // clean miss 
            if (nHitPopulation == 0)
                continue;

            Sse.Prefetch1(pN->GetPrefetch());

            // skip ray reshuffling on a clean hit
            if (nHitPopulation < pS->nRayPop)
            {
                nGroups = (uint)RemoveMissedGroups(frame.pActivePackets, (byte*)Unsafe.AsPointer(ref frame.pMasks[0]), (int)nGroups);
#if !NO_REORDERING
                // re-sort incoherent packets if coherency is low enough
                if (nGroups > 1 && (8 * nGroups - nHitPopulation) >= 4 * nGroups) // 50% utilization
                {
                    ReorderRays(ref frame, nGroups);

                    // packets are now fully coherent.  There is at most one underutilized packet.
                    //  Reduce the number of active groups and append reordered packets onto end of group list
                    nGroups = RoundUp8((uint)nHitPopulation) / 8;
                }
#endif
            }

            if (pN->IsLeaf())
            {
                // visit leaf
                fixed (StackFrame* pFrame = &frame)
                {
                    for (int g = 0; g < nGroups; g++)
                        TriISectPacket_Preproc_List(pN->GetTriList(), frame.pActivePackets[g], pFrame);
                }
            }
            else if (nGroups <= 1)
            {
                // if we're down to a single group, dispatch single packet traversal 

                for (int g = 0; g < nGroups; g++)
                {
                    PacketTrace_Octant(ref *frame.pActivePackets[g].Value, nRayOctant, pStack, pN, ref frame);
                }
            }
            else
            {
                // push subtrees in correct order
                ulong axis = pN->GetSplitAxis();
                long lf = ((long)nRayOctant >> (int)axis) & 1; // ray dir is negative -->  visit left first --> push right first

                BVHNode* pSecond = pN->GetLeftChild() + (lf ^ 1);
                BVHNode* pFirst = pN->GetLeftChild() + lf;
                pStack[0].nGroups = nGroups;
                pStack[0].pN = pSecond;
                pStack[0].nRayPop = nHitPopulation;
                pStack[1].nGroups = nGroups;
                pStack[1].pN = pFirst;
                pStack[1].nRayPop = nHitPopulation;
                pStack += 2;
            }
        }
    }
}

