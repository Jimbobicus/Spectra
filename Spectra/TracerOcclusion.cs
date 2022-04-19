using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

using static Spectra.Raytracer;
using static System.Runtime.Intrinsics.X86.Avx;
using static System.Runtime.Intrinsics.X86.Avx2;
using static System.Runtime.Intrinsics.X86.Fma;

namespace Spectra;

unsafe partial struct Tracer
{
    static long TriISectPacket_Preproc_List_Occ(long mask, TriList* pList, RayPacket* pPacket, StackFrame* pFrame)
    {
        Vector256<float> h = Vector256<float>.Zero;
        Vector256<float> TMax = pPacket->TMax;

        Vector256<float>* D = stackalloc Vector256<float>[3];
        ReadDirs(D, pPacket, pFrame->pRays);

        long m = mask;
        long nTris = pList->GetTriCount();
        for (long i = 0; i < nTris; i++)
        {
            PreprocessedTri* pTri = pList->GetTriList() + i;
            float* P0 = pTri->P0;
            float* v10 = pTri->v10;
            float* v02 = pTri->v02;
            float* v10x02 = pTri->v10x02;

            Sse.PrefetchNonTemporal((char*)(pTri + 1));

            Vector256<float> vDx = D[0];
            Vector256<float> vDy = D[1];
            Vector256<float> vDz = D[2];
            Vector256<float>[] v0A = new[] {
                Subtract( BroadcastScalarToVector256( &P0[0] ), pPacket->Ox ),
                Subtract( BroadcastScalarToVector256( &P0[1] ), pPacket->Oy ),
                Subtract( BroadcastScalarToVector256( &P0[2] ), pPacket->Oz ),
            };
            Vector256<float>[] v0AxD = new[]{
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

            inside = And(inside, CompareLessThan(Add(A, B), Vector256.Create(1.0f)));

            Vector256<float> hit = And(inside, front);
            h = Or(h, hit);

            m = MoveMask(h);
            if (m == 0xff)
                break;
        }

        // force TMax=-1 for hit rays so that all subsequent intersection tests on them will miss
        Vector256<float> MINUS_ONE = BroadcastScalarToVector256(Vector128.Create(-1.0f));
        TMax = BlendVariable(TMax, MINUS_ONE, h);
        StoreAligned((float*)&(pPacket->TMax), TMax);

        // mark hits in output array and set t=0 in ray structs
        ulong k = (ulong)m;
        ulong idx = Bmi1.X64.TrailingZeroCount(k);
        while (idx < 64)
        {
            long rid = pPacket->RayOffsets[idx] / sizeof(Ray);
            k = Bmi1.X64.ResetLowestSetBit(k);
            idx = Bmi1.X64.TrailingZeroCount(k);
            pFrame->pOcclusion[rid] = 0xff;
            pFrame->pRays[rid].tmax = -1.0f;
        }


        return m;
    }


    static long PacketTrace_Octant_Occ(ref RayPacket pack, long mask, uint octant, void* pStackBottom, BVHNode* pRoot, ref StackFrame frame)
    {

        BVHNode** pStack = (BVHNode**)pStackBottom;
        *(pStack++) = pRoot;

        // positive sign:  first=0, last=3
        // negative sign:  first=3, last=0
        uint xfirst = ((octant & 1) != 0) ? 3u : 0;
        uint yfirst = ((octant & 2) != 0) ? 3u : 0;
        uint zfirst = ((octant & 4) != 0) ? 3u : 0;


        var OD = new[] {
            Multiply( pack.DInvx, pack.Ox ),
            Multiply( pack.DInvy, pack.Oy ),
            Multiply( pack.DInvz, pack.Oz ),
        };

        // pick min/max if octant sign is zero, reverse otherwise
        var vOctant = Vector128.Create(0, ((octant & 1) != 0) ? 0xffffffff : 0,
                                          ((octant & 2) != 0) ? 0xffffffff : 0,
                                          ((octant & 4) != 0) ? 0xffffffff : 0);
        Vector256<float> octant_select = BroadcastVector128ToVector256((int*)Unsafe.AsPointer(ref vOctant)).AsSingle();


        while (pStack != pStackBottom && mask != 0xff)
        {
            BVHNode* pN = *(--pStack);
            if (pN->IsLeaf())
            {
                // visit leaf
                TriList* pList = pN->GetTriList();
                fixed (RayPacket* pPacket = &pack)
                fixed (StackFrame* pFrame = &frame)
                    mask |= TriISectPacket_Preproc_List_Occ(mask, pList, pPacket, pFrame);
            }
            else
            {
                BVHNode* pNLeft = pN->GetLeftChild();
                BVHNode* pNRight = pN->GetRightChild();
                float* pLeft = (float*)pNLeft->m_AABB;
                float* pRight = (float*)pNRight->m_AABB;
                Sse.Prefetch0((char*)pNLeft->GetPrefetch());
                Sse.Prefetch0((char*)pNRight->GetPrefetch());

                /*
                // test children, push far ones, descend to near ones
                Vector256<float> Bmin0  = BroadcastScalarToVector256( pLeft  + xfirst      );
                Vector256<float> Bmax0  = BroadcastScalarToVector256( pLeft  + (xfirst^3)  );
                Vector256<float> Bmin1  = BroadcastScalarToVector256( pRight + xfirst      );
                Vector256<float> Bmax1  = BroadcastScalarToVector256( pRight + (xfirst^3)  );
                Vector256<float> tmin0 = Multiply( _mm256_sub_ps( Bmin0, pack.Ox ),pack.DInvx );
                Vector256<float> tmax0 = Multiply( _mm256_sub_ps( Bmax0, pack.Ox ),pack.DInvx );
                Vector256<float> tmin1 = Multiply( _mm256_sub_ps( Bmin1, pack.Ox ),pack.DInvx );
                Vector256<float> tmax1 = Multiply( _mm256_sub_ps( Bmax1, pack.Ox ),pack.DInvx );
                                                 
                Bmin0  = BroadcastScalarToVector256( pLeft  + 1 + yfirst      );
                Bmax0  = BroadcastScalarToVector256( pLeft  + 1 + (yfirst^3)  );
                Bmin1  = BroadcastScalarToVector256( pRight + 1 + yfirst      );
                Bmax1  = BroadcastScalarToVector256( pRight + 1 + (yfirst^3)  );
                tmin0  = Max( tmin0, Multiply( _mm256_sub_ps( Bmin0, pack.Oy ),pack.DInvy  ) );
                tmax0  = Min( tmax0, Multiply( _mm256_sub_ps( Bmax0, pack.Oy ),pack.DInvy  ) );
                tmin1  = Max( tmin1, Multiply( _mm256_sub_ps( Bmin1, pack.Oy ),pack.DInvy  ) );
                tmax1  = Min( tmax1, Multiply( _mm256_sub_ps( Bmax1, pack.Oy ),pack.DInvy  ) );
    
                Bmin0  = BroadcastScalarToVector256( pLeft  + 2 + zfirst     );
                Bmax0  = BroadcastScalarToVector256( pLeft  + 2 + (zfirst^3) );
                Bmin1  = BroadcastScalarToVector256( pRight + 2 + zfirst     );
                Bmax1  = BroadcastScalarToVector256( pRight + 2 + (zfirst^3) );
                tmin0  = Max( tmin0, Multiply( _mm256_sub_ps( Bmin0, pack.Oz ),pack.DInvz ) );
                tmax0  = Min( tmax0, Multiply( _mm256_sub_ps( Bmax0, pack.Oz ),pack.DInvz ) );
                tmin1  = Max( tmin1, Multiply( _mm256_sub_ps( Bmin1, pack.Oz ),pack.DInvz ) );
                tmax1  = Min( tmax1, Multiply( _mm256_sub_ps( Bmax1, pack.Oz ),pack.DInvz ) );
            */

                /*
                // test children, push far ones, descend to near ones
                Vector256<float> Bmin0  = BroadcastScalarToVector256( pLeft  + xfirst      );
                Vector256<float> Bmax0  = BroadcastScalarToVector256( pLeft  + (xfirst^3)  );
                Vector256<float> Bmin1  = BroadcastScalarToVector256( pRight + xfirst      );
                Vector256<float> Bmax1  = BroadcastScalarToVector256( pRight + (xfirst^3)  );
               
                Vector256<float> tmin0 = MultiplySubtract( Bmin0, pack.DInvx, rx );
                Vector256<float> tmax0 = MultiplySubtract( Bmax0, pack.DInvx, rx );
                Vector256<float> tmin1 = MultiplySubtract( Bmin1, pack.DInvx, rx );
                Vector256<float> tmax1 = MultiplySubtract( Bmax1, pack.DInvx, rx );
                Bmin0  = BroadcastScalarToVector256( pLeft  + 1 + yfirst      );
                Bmax0  = BroadcastScalarToVector256( pLeft  + 1 + (yfirst^3)  );
                Bmin1  = BroadcastScalarToVector256( pRight + 1 + yfirst      );
                Bmax1  = BroadcastScalarToVector256( pRight + 1 + (yfirst^3)  );
                tmin0 = Max( tmin0, MultiplySubtract( Bmin0, pack.DInvy, ry ) );
                tmax0 = Min( tmax0, MultiplySubtract( Bmax0, pack.DInvy, ry ) );
                tmin1 = Max( tmin1, MultiplySubtract( Bmin1, pack.DInvy, ry ) );
                tmax1 = Min( tmax1, MultiplySubtract( Bmax1, pack.DInvy, ry ) );
                Bmin0  = BroadcastScalarToVector256( pLeft  + 2 + zfirst     );
                Bmax0  = BroadcastScalarToVector256( pLeft  + 2 + (zfirst^3) );
                Bmin1  = BroadcastScalarToVector256( pRight + 2 + zfirst     );
                Bmax1  = BroadcastScalarToVector256( pRight + 2 + (zfirst^3) );
                tmin0 = Max( tmin0, MultiplySubtract( Bmin0, pack.DInvz, rz ) );
                tmax0 = Min( tmax0, MultiplySubtract( Bmax0, pack.DInvz, rz ) );
                tmin1 = Max( tmin1, MultiplySubtract( Bmin1, pack.DInvz, rz ) );
                tmax1 = Min( tmax1, MultiplySubtract( Bmax1, pack.DInvz, rz ) );
                */



                // fetch AABB
                // bbmin(xyz), bbmax(xyz)
                BVHNode* pL = pN->GetLeftChild();
                BVHNode* pR = pN->GetRightChild();
                Vector256<float> min0 = BroadcastScalarToVector256((pLeft));
                Vector256<float> min1 = BroadcastScalarToVector256((pRight));
                Vector256<float> max0 = BroadcastScalarToVector256((pLeft + 3));
                Vector256<float> max1 = BroadcastScalarToVector256((pRight + 3));

                // swap planes based on octant
                Vector256<float> bbmin0 = BlendVariable(min0, max0, octant_select);
                Vector256<float> bbmax0 = BlendVariable(max0, min0, octant_select);
                Vector256<float> bbmin1 = BlendVariable(min1, max1, octant_select);
                Vector256<float> bbmax1 = BlendVariable(max1, min1, octant_select);

                Sse.Prefetch0((char*)pNLeft->GetPrefetch());
                Sse.Prefetch0((char*)pNRight->GetPrefetch());


                // axis tests
                Vector256<float> D = pack.DInvx;
                Vector256<float> O = OD[0];
                Vector256<float> Bmin0 = Permute(bbmin0, 0x00);
                Vector256<float> Bmax0 = Permute(bbmax0, 0x00);
                Vector256<float> Bmin1 = Permute(bbmin1, 0x00);
                Vector256<float> Bmax1 = Permute(bbmax1, 0x00);
                Vector256<float> tmin0 = MultiplySubtract(Bmin0, D, O);
                Vector256<float> tmax0 = MultiplySubtract(Bmax0, D, O);
                Vector256<float> tmin1 = MultiplySubtract(Bmin1, D, O);
                Vector256<float> tmax1 = MultiplySubtract(Bmax1, D, O);

                D = pack.DInvy;
                O = OD[1];
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

                D = pack.DInvz;
                O = OD[2];
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
                Vector256<float> hitL = AndNot(tmax0, CompareLessThanOrEqual(tmin0, limL));
                Vector256<float> hitR = AndNot(tmax1, CompareLessThanOrEqual(tmin1, limR));


                int maskhitL = MoveMask(hitL);
                int maskhitR = MoveMask(hitR);
                int maskhitB = maskhitL & maskhitR;

                if (maskhitR != 0)
                    *(pStack++) = pNRight;
                if (maskhitL != 0)
                    *(pStack++) = pNLeft;
            }
        }

        return mask;
    }


    static long ReadOcclusionMask(RayPacket* pPack, byte* pOcclusion)
    {
        long m0 = pOcclusion[pPack->RayOffsets[0] / sizeof(Ray)] & 1;
        long m1 = pOcclusion[pPack->RayOffsets[1] / sizeof(Ray)] & 2;
        long m2 = pOcclusion[pPack->RayOffsets[2] / sizeof(Ray)] & 4;
        long m3 = pOcclusion[pPack->RayOffsets[3] / sizeof(Ray)] & 8;
        long m4 = pOcclusion[pPack->RayOffsets[4] / sizeof(Ray)] & 16;
        long m5 = pOcclusion[pPack->RayOffsets[5] / sizeof(Ray)] & 32;
        long m6 = pOcclusion[pPack->RayOffsets[6] / sizeof(Ray)] & 64;
        long m7 = pOcclusion[pPack->RayOffsets[7] / sizeof(Ray)] & 128;
        return (m0 | m1) | (m2 | m3) | (m4 | m5) | (m6 | m7);
    }



    unsafe struct Stack
    {
        public BVHNode* pN;
        public long nGroups;
        public long nRayPop;
    }


    void AdaptiveTrace_Occlusion(void* pStackMem, ref StackFrame frame, uint nRayOctant)
    {
        BVHNode* pBVH = frame.pBVH;
        RayPacket* pPackets = frame.pAllPackets;

        uint nPackets = (uint)frame.nPackets;
        for (int i = 0; i < nPackets; i++)
            frame.pActivePackets[i] = frame.pAllPackets + i;

        Stack* pStackBottom = (Stack*)pStackMem;
        Stack* pStack = pStackBottom;
        pStack->nGroups = nPackets;
        pStack->pN = pBVH;
        pStack->nRayPop = 8 * nPackets;
        ++pStack;


        long xfirst = (nRayOctant & 1) != 0 ? 3 : 0;
        long yfirst = (nRayOctant & 2) != 0 ? 3 : 0;
        long zfirst = (nRayOctant & 4) != 0 ? 3 : 0;

        while (pStack != pStackBottom)
        {
            Stack* pS = (--pStack);
            BVHNode* pN = pS->pN;
            long nGroups = pS->nGroups;
            if (nGroups == 0)
                continue;

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
                fixed (byte* pMasks = frame.pMasks)
                    nGroups = RemoveMissedGroups(frame.pActivePackets, pMasks, (int)nGroups);

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
                int nOut = 0;

                for (int g = 0; g < nGroups; g++)
                {
                    RayPacket* p = frame.pActivePackets[g];
                    long m = ReadOcclusionMask(p, frame.pOcclusion);
                    fixed (StackFrame* pFrame = &frame)
                        TriISectPacket_Preproc_List_Occ(m, pN->GetTriList(), p, pFrame);

                    if (m != 0xff)
                        frame.pActivePackets[nOut++] = p;
                }

                if (nOut < nGroups)
                {
                    // some packets were completely occluded and dropped out.  Shift the rest of the active packets array backwards
                    int nDropped = (int)(nGroups - nOut);
                    for (int g = (int)nGroups; g < frame.nPackets; g++)
                        frame.pActivePackets[g - nDropped] = frame.pActivePackets[g];
                    frame.nPackets -= nDropped;

                    // adjust stack entries to make it look like the occluded packets never existed
                    for (Stack* s = pStackBottom; s != pStack; s++)
                    {
                        s->nGroups -= nDropped;
                        s->nRayPop -= 8 * nDropped;
                    }
                }

            }
            else if (nGroups == 1)
            {
                // if we're down to a single group, dispatch single packet traversal 
                RayPacket* p = frame.pActivePackets[0];
                long m = ReadOcclusionMask(p, frame.pOcclusion);
                PacketTrace_Octant_Occ(ref *p, m, nRayOctant, pStack, pN, ref frame);

                if (m == 0xff)
                {
                    // this packet is dead.  Shift it out
                    for (int g = 1; g < frame.nPackets; g++)
                        frame.pActivePackets[g - 1] = frame.pActivePackets[g];
                    frame.nPackets--;

                    // adjust stack entries to make it look like the occluded packets never existed
                    for (Stack* s = pStackBottom; s != pStack; s++)
                    {
                        s->nGroups -= 1;
                        s->nRayPop -= 8;
                    }
                }

            }
            else
            {
                // left, then right
                pStack[0].nGroups = nGroups;
                pStack[0].pN = pN->GetRightChild();
                pStack[0].nRayPop = nHitPopulation;
                pStack[1].nGroups = nGroups;
                pStack[1].pN = pN->GetLeftChild();
                pStack[1].nRayPop = nHitPopulation;
                pStack += 2;
            }
        }
    }


    void DoOcclusionTrace(ref Tracer pTracer, byte* pOcclusionOut)
    {
        RayPacket* packs = stackalloc RayPacket[MAX_TRACER_SIZE / 8 + 8]; // up to one additional packet per octant

        StackFrame frame;
        frame.pRays = pTracer.pRays;
        frame.nRays = pTracer.nRays;
        frame.pAllPackets = packs;
        frame.pHitInfo = null;
        frame.pOcclusion = pOcclusionOut;
        frame.pBVH = pTracer.pBVHRoot;

        for (long i = 0; i < frame.nRays; i++)
            pOcclusionOut[i] = 0;

        uint* pPacksByOctant = stackalloc uint[8];
        BuildPacketsByOctant(packs, ref pTracer, pPacksByOctant);

        RayPacket* pPacket = packs;
        for (uint i = 0; i < 8; i++)
        {
            long n = pPacksByOctant[i];
            if (n != 0)
            {
                frame.nPackets = n;
                AdaptiveTrace_Occlusion(pTracer.pTraversalStack, ref frame, i);
            }

            frame.pAllPackets += n;
        }
    }
}
