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

[StructLayout(LayoutKind.Sequential, Pack = 64)]
unsafe struct RayPacket
{
    public Vector256<float> Ox, DInvx;  // NOTE: Order matters.  Each line here is 1 cacheline
    public Vector256<float> Oy, DInvy;  // The O/DInv pairs are always read together during traversal
    public Vector256<float> Oz, DInvz;
    public Vector256<float> TMax; public fixed uint RayOffsets[8];
}

unsafe ref struct StackFrame
{
    public BVHNode* pBVH;
    public Ray* pRays;
    public long nRays;
    public long nPackets;

    public RayPacket* pAllPackets;
    public RayHitInfo* pHitInfo;
    public byte* pOcclusion;
    public fixed float pAABB[6];
    ///<  Indices of active packets in pPackets 
    public Span<Pointer<RayPacket>> pActivePackets =>
        MemoryMarshal.Cast<ulong, Pointer<RayPacket>>(MemoryMarshal.CreateSpan(ref ppActivePackets[0], Tracer.MAX_PACKETS_IN_FLIGHT));
    public fixed byte pMasks[Tracer.MAX_PACKETS_IN_FLIGHT];        ///< Ray mask for each active packet
                                                                   ///<  This is indexed over the packets, not the active groups in pGroupIDs
    fixed ulong ppActivePackets[Tracer.MAX_PACKETS_IN_FLIGHT]; // (RayPacket*)
}
unsafe partial struct Tracer
{
    internal const int MAX_PACKETS_IN_FLIGHT = Raytracer.MAX_TRACER_SIZE / 8;

    static uint RoundUp8(uint n)
    {
        return (n + 7u) & ~7u;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> RCPNR(Vector256<float> f)
    {
        Vector256<float> rcp = Reciprocal(f);
        Vector256<float> rcp_sq = Multiply(rcp, rcp);
        Vector256<float> rcp_x2 = Add(rcp, rcp);
        return MultiplyAddNegated(rcp_sq, f, rcp_x2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<int> BROADCASTINT(long x)
    {
        return BroadcastScalarToVector256(Sse2.ConvertScalarToVector128Int32((int)x));
    }


    static byte SHUFFLE(long a, long b, long c, long d) => (byte)(a | (b << 2) | (c << 4) | (d << 6));

    static void ReadDirs(Vector256<float>* D, RayPacket* pPacket, Ray* pRays)
    {
#if GATHERS
        Vector256<int> idx = _mm256_load_si256((Vector256<int>*)pPacket->RayOffsets);
        Vector256<float> Dx = _mm256_i32gather_ps( &pRays->dx, idx, 1 );
        Vector256<float> Dy = _mm256_i32gather_ps( &pRays->dy, idx, 1 );
        Vector256<float> Dz = _mm256_i32gather_ps( &pRays->dz, idx, 1 );
        D[0] = Dx;
        D[1] = Dy;
        D[2] = Dz;
#else
        byte* pBytes = ((byte*)pRays) + 16;
        Vector128<float> LOADPS(void* x) => Sse.LoadAlignedVector128((float*)(x));
        var v0 = Vector256.Create(LOADPS(pBytes + pPacket->RayOffsets[4]),
                                      LOADPS(pBytes + pPacket->RayOffsets[0]));  // 0000 4444
        var v1 = Vector256.Create(LOADPS(pBytes + pPacket->RayOffsets[6]),
                                      LOADPS(pBytes + pPacket->RayOffsets[2]));  // 2222 6666
        var v2 = Vector256.Create(LOADPS(pBytes + pPacket->RayOffsets[5]),
                                      LOADPS(pBytes + pPacket->RayOffsets[1]));  // 1111 5555
        var v3 = Vector256.Create(LOADPS(pBytes + pPacket->RayOffsets[7]),
                                      LOADPS(pBytes + pPacket->RayOffsets[3]));  // 3333 7777

        var t0 = UnpackLow(v0, v1); // 02 02 46 46
        var t1 = UnpackHigh(v0, v1); // 02 02 46 46
        var t2 = UnpackLow(v2, v3); // 13 13 57 57
        var t3 = UnpackHigh(v2, v3); // 13 13 57 57
        var X = UnpackLow(t0, t2);  // 01 23 45 67
        var Y = UnpackHigh(t0, t2);
        var Z = UnpackLow(t1, t3);
        D[0] = X;
        D[1] = Y;
        D[2] = Z;
#endif
    }

    static void ReadOrigins(Span<Vector256<float>> O, RayPacket* pPacket, Ray* pRays)
    {
#if GATHERS
        Vector256<int> idx = _mm256_load_si256((Vector256<int>*)pPacket->RayOffsets);
        Vector256<float> Ox = _mm256_i32gather_ps( &pRays->ox, idx, 1 );
        Vector256<float> Oy = _mm256_i32gather_ps( &pRays->oy, idx, 1 );
        Vector256<float> Oz = _mm256_i32gather_ps( &pRays->oz, idx, 1 );
        O[0] = Ox;
        O[1] = Oy;
        O[2] = Oz;
#else
        byte* pBytes = ((byte*)pRays);
        Vector128<float> LOADPS(void* x) => Sse.LoadAlignedVector128((float*)(x));
        var v0 = Vector256.Create(LOADPS(pBytes + pPacket->RayOffsets[4]),
                                      LOADPS(pBytes + pPacket->RayOffsets[0]));  // 0000 4444
        var v1 = Vector256.Create(LOADPS(pBytes + pPacket->RayOffsets[6]),
                                      LOADPS(pBytes + pPacket->RayOffsets[2]));  // 2222 6666
        var v2 = Vector256.Create(LOADPS(pBytes + pPacket->RayOffsets[5]),
                                      LOADPS(pBytes + pPacket->RayOffsets[1]));  // 1111 5555
        var v3 = Vector256.Create(LOADPS(pBytes + pPacket->RayOffsets[7]),
                                      LOADPS(pBytes + pPacket->RayOffsets[3]));  // 3333 7777

        Vector256<float> t0 = UnpackLow(v0, v1); // 02 02 46 46
        Vector256<float> t1 = UnpackHigh(v0, v1); // 02 02 46 46
        Vector256<float> t2 = UnpackLow(v2, v3); // 13 13 57 57
        Vector256<float> t3 = UnpackHigh(v2, v3); // 13 13 57 57
        Vector256<float> X = UnpackLow(t0, t2);  // 01 23 45 67
        Vector256<float> Y = UnpackHigh(t0, t2);
        Vector256<float> Z = UnpackLow(t1, t3);
        O[0] = X;
        O[1] = Y;
        O[2] = Z;
#endif
    }


    static void ReadRays(RayPacket* pPacket, byte* pRays, uint* pOffsets)
    {

#if GATHERS
        Ray* pR = (Ray*)pRays;
        Vector256<int> idx = _mm256_loadu_si256((Vector256<int>*)pOffsets);
        Vector256<float> Ox   = _mm256_i32gather_ps( &pR->ox, idx, 1 );
        Vector256<float> Oy   = _mm256_i32gather_ps( &pR->oy, idx, 1 );
        Vector256<float> Oz   = _mm256_i32gather_ps( &pR->oz, idx, 1 );
        Vector256<float> Tmax = _mm256_i32gather_ps( &pR->tmax, idx, 1 );
        Vector256<float> Dx   = _mm256_i32gather_ps( &pR->dx, idx, 1 );
        Vector256<float> Dy   = _mm256_i32gather_ps( &pR->dy, idx, 1 );
        Vector256<float> Dz   = _mm256_i32gather_ps( &pR->dz, idx, 1 );
        Vector256<float> RID  = _mm256_i32gather_ps( (float*)&(pR->offset), idx,1 );
        Dx = RCPNR(Dx);
        Dy = RCPNR(Dy);
        Dz = RCPNR(Dz);
        pPacket->Ox = Ox;
        pPacket->Oy = Oy;
        pPacket->Oz = Oz;
        pPacket->DInvx = Dx;
        pPacket->DInvy = Dy;
        pPacket->DInvz = Dz;
        pPacket->TMax = Tmax;
        StoreAligned((float*)(pPacket->RayOffsets), RID );
#else
        // unpacklo(x,y) -->   x0 y0 x1 y1 x4 y4 x5 y5
        // 0 1 2 3   0 1 2 3
        //  
        //   0 1 2 3   0 1 2 3
        //   0 1 2 3   0 1 2 3
        //   0 1 2 3   0 1 2 3
        //   0 1 2 3   0 1 2 3

        Vector128<float> LOADPS(void* x) => Sse.LoadAlignedVector128((float*)(x));
        Vector256<float> l0 = Vector256.Create(LOADPS(pRays + pOffsets[1]), LOADPS(pRays + pOffsets[0]));
        Vector256<float> l1 = Vector256.Create(LOADPS(pRays + pOffsets[3]), LOADPS(pRays + pOffsets[2]));
        Vector256<float> l2 = Vector256.Create(LOADPS(pRays + pOffsets[5]), LOADPS(pRays + pOffsets[4]));
        Vector256<float> l3 = Vector256.Create(LOADPS(pRays + pOffsets[7]), LOADPS(pRays + pOffsets[6]));
        Vector256<float> l4 = Vector256.Create(LOADPS(pRays + pOffsets[1] + 16), LOADPS(pRays + pOffsets[0] + 16));
        Vector256<float> l5 = Vector256.Create(LOADPS(pRays + pOffsets[3] + 16), LOADPS(pRays + pOffsets[2] + 16));
        Vector256<float> l6 = Vector256.Create(LOADPS(pRays + pOffsets[5] + 16), LOADPS(pRays + pOffsets[4] + 16));
        Vector256<float> l7 = Vector256.Create(LOADPS(pRays + pOffsets[7] + 16), LOADPS(pRays + pOffsets[6] + 16));

        Vector256<float> t0 = UnpackLow(l0, l1); // 00 11 00 11
        Vector256<float> t1 = UnpackLow(l2, l3); // 00 11 00 11
        Vector256<float> t2 = UnpackHigh(l0, l1); // 22 33 22 33
        Vector256<float> t3 = UnpackHigh(l2, l3); // 22 33 22 33
        Vector256<float> t4 = UnpackLow(l4, l5); // 44 55 44 55
        Vector256<float> t5 = UnpackLow(l6, l7); // 44 55 44 55
        Vector256<float> t6 = UnpackHigh(l4, l5); // 66 77 66 77 
        Vector256<float> t7 = UnpackHigh(l6, l7); // 66 77 66 77

        Vector256<float> Ox = UnpackLow(t0, t1); // 00 00 00 00
        Vector256<float> Oy = UnpackHigh(t0, t1); // 11 11 11 11
        Vector256<float> Oz = UnpackLow(t2, t3); // 22 22 22 22
        Vector256<float> TMax = UnpackHigh(t2, t3); // 33 33 33 33
        Vector256<float> Dx = UnpackLow(t4, t5);
        Vector256<float> Dy = UnpackHigh(t4, t5);
        Vector256<float> Dz = UnpackLow(t6, t7);
        Vector256<float> RID = UnpackHigh(t6, t7);

        StoreAligned((float*)&pPacket->Ox, Ox);
        StoreAligned((float*)&pPacket->DInvx, RCPNR(Dx));
        StoreAligned((float*)&pPacket->Oy, Oy);
        StoreAligned((float*)&pPacket->DInvy, RCPNR(Dy));
        StoreAligned((float*)&pPacket->Oz, Oz);
        StoreAligned((float*)&pPacket->DInvz, RCPNR(Dz));
        StoreAligned((float*)&pPacket->TMax, TMax);
        StoreAligned((float*)pPacket->RayOffsets, RID);

        // unpacklo(x,y) -->   x0 y0 x1 y1 x4 y4 x5 y5
        // 0 1 2 3   0 1 2 3
        //  
        //   0 1 2 3   0 1 2 3
        //   0 1 2 3   0 1 2 3
        //   0 1 2 3   0 1 2 3
        //   0 1 2 3   0 1 2 3
#endif
    }



    unsafe struct ReadRaysLoopArgs
    {
        public uint* pRayIDs;
        public RayPacket** pPackets;
        public byte* pRays;
    };

    static void ReadRaysLoop(ref ReadRaysLoopArgs l, long nReorder)
    {
        byte* pRays = l.pRays;
        for (long i = 0; i < nReorder; i++)
        {
            uint* pOffsets = l.pRayIDs + 8 * i;

#if GATHERS
        Ray* pR = (Ray*)pRays;
         Vector256<int> idx = _mm256_loadu_si256((Vector256<int>*)pOffsets);
        Vector256<float> Ox   = _mm256_i32gather_ps( &pR->ox, idx, 1 );
        Vector256<float> Oy   = _mm256_i32gather_ps( &pR->oy, idx, 1 );
        Vector256<float> Oz   = _mm256_i32gather_ps( &pR->oz, idx, 1 );
        Vector256<float> Tmax = _mm256_i32gather_ps( &pR->tmax, idx, 1 );
        Vector256<float> Dx   = _mm256_i32gather_ps( &pR->dx, idx, 1 );
        Vector256<float> Dy   = _mm256_i32gather_ps( &pR->dy, idx, 1 );
        Vector256<float> Dz   = _mm256_i32gather_ps( &pR->dz, idx, 1 );
        Vector256<float> RID  = _mm256_i32gather_ps( (float*)&(pR->offset), idx,1 );
        Dx = RCPNR(Dx);
        Dy = RCPNR(Dy);
        Dz = RCPNR(Dz);
        
        RayPacket* __restrict pPacket = l.pPackets[i];
        pPacket->Ox = Ox;
        pPacket->Oy = Oy;
        pPacket->Oz = Oz;
        pPacket->DInvx = Dx;
        pPacket->DInvy = Dy;
        pPacket->DInvz = Dz;
        pPacket->TMax = Tmax;
        StoreAligned((float*)(pPacket->RayOffsets), RID );

#else

            // Load lower halves into L0-L3, and upper halves into L4-L7
            //   Lower half contains origin (x,y,z) and TMax
            //   Upper half contains directions(x,y,z), and byte offset from start of ray stream
            //
            // Using 128-bit loads and inserts is preferable to 256-bit loads and cross-permutes
            //  The inserts can be fused with the loads, and Haswell can issue them on more ports that way
            Vector128<float> LOADPS(void* x) => Sse.LoadAlignedVector128((float*)(x));
            Vector256<float> l0 = Vector256.Create(LOADPS(pRays + pOffsets[1]), LOADPS(pRays + pOffsets[0]));
            Vector256<float> l1 = Vector256.Create(LOADPS(pRays + pOffsets[3]), LOADPS(pRays + pOffsets[2]));
            Vector256<float> l2 = Vector256.Create(LOADPS(pRays + pOffsets[5]), LOADPS(pRays + pOffsets[4]));
            Vector256<float> l3 = Vector256.Create(LOADPS(pRays + pOffsets[7]), LOADPS(pRays + pOffsets[6]));
            Vector256<float> l4 = Vector256.Create(LOADPS(pRays + pOffsets[1] + 16), LOADPS(pRays + pOffsets[0] + 16));
            Vector256<float> l5 = Vector256.Create(LOADPS(pRays + pOffsets[3] + 16), LOADPS(pRays + pOffsets[2] + 16));
            Vector256<float> l6 = Vector256.Create(LOADPS(pRays + pOffsets[5] + 16), LOADPS(pRays + pOffsets[4] + 16));
            Vector256<float> l7 = Vector256.Create(LOADPS(pRays + pOffsets[7] + 16), LOADPS(pRays + pOffsets[6] + 16));

            Vector256<float> t4 = UnpackLow(l4, l5); // 44 55 44 55
            Vector256<float> t5 = UnpackLow(l6, l7); // 44 55 44 55       
            t4 = RCPNR(t4); // both 4 and 5 get rcp'd eventually, so we can start them earlier
            t5 = RCPNR(t5); // to give the other ports something to do during this monstrous blob of unpacks

            Vector256<float> t0 = UnpackLow(l0, l1); // 00 11 00 11
            Vector256<float> t1 = UnpackLow(l2, l3); // 00 11 00 11
            Vector256<float> t2 = UnpackHigh(l0, l1); // 22 33 22 33
            Vector256<float> t3 = UnpackHigh(l2, l3); // 22 33 22 33
            Vector256<float> t6 = UnpackHigh(l4, l5); // 66 77 66 77 
            Vector256<float> t7 = UnpackHigh(l6, l7); // 66 77 66 77
            Vector256<float> Ox = UnpackLow(t0, t1); // 00 00 00 00
            Vector256<float> Oy = UnpackHigh(t0, t1); // 11 11 11 11
            Vector256<float> Oz = UnpackLow(t2, t3); // 22 22 22 22
            Vector256<float> TMax = UnpackHigh(t2, t3); // 33 33 33 33
            Vector256<float> Dx = UnpackLow(t4, t5);
            Vector256<float> Dy = UnpackHigh(t4, t5);
            Vector256<float> Dz = UnpackLow(t6, t7);
            Vector256<float> RID = UnpackHigh(t6, t7);

            RayPacket* pPacket = l.pPackets[i];
            StoreAligned((float*)&pPacket->Ox, Ox);
            StoreAligned((float*)&pPacket->DInvx, (Dx));
            StoreAligned((float*)&pPacket->Oy, Oy);
            StoreAligned((float*)&pPacket->DInvy, (Dy));
            StoreAligned((float*)&pPacket->Oz, Oz);
            StoreAligned((float*)&pPacket->DInvz, RCPNR(Dz));
            StoreAligned((float*)&pPacket->TMax, TMax);
            StoreAligned((float*)pPacket->RayOffsets, RID);
#endif
        }
    }


    static long GroupTest2X(StackFrame* pFrame, long nGroups)
    {
        RayPacket** pPackets = (RayPacket**)Unsafe.AsPointer(ref pFrame->pActivePackets[0]);
        byte* pMasks = pFrame->pMasks;
        float* pAABB = pFrame->pAABB;

        long g;
        long nTwos = (nGroups & ~1);
        long nHitPopulation = 0;
        for (g = 0; g < nTwos; g += 2)
        {
            ref RayPacket pack0 = ref *pPackets[g];
            ref RayPacket pack1 = ref *pPackets[g + 1];

            ///////////////////////////////////////////////////////////////////////////////////
            // VERSION 1:  sub, mul.    
            //  MSVC likes to spill unless you spell things out for out very explicitly...
            ///////////////////////////////////////////////////////////////////////////////////
            // Vector256<float> Bmin  = BroadcastScalarToVector256( pAABB+0 );
            // Vector256<float> Bmax  = BroadcastScalarToVector256( pAABB+3 );
            //  Vector256<float> O0 = Vector256.Create( (float*)&pack0.Ox );
            //  Vector256<float> O1 = Vector256.Create( (float*)&pack1.Ox );
            //  Vector256<float> D0 = Vector256.Create( (float*)&pack0.DInvx );
            //  Vector256<float> D1 = Vector256.Create( (float*)&pack1.DInvx );
            // Vector256<float> t0 =  Subtract( Bmin, O0 );
            // Vector256<float> t1 =  Subtract( Bmax, O0 );
            // Vector256<float> t2 =  Subtract( Bmin, O1 );
            // Vector256<float> t3 =  Subtract( Bmax, O1 );
            // Vector256<float> tmin0 = Multiply(t0,D0 );
            // Vector256<float> tmax0 = Multiply(t1,D0 );
            // Vector256<float> tmin1 = Multiply(t2,D1 );
            // Vector256<float> tmax1 = Multiply(t3,D1 );
            // Bmin   = BroadcastScalarToVector256( pAABB+1 );
            // Bmax   = BroadcastScalarToVector256( pAABB+4 );
            // 
            // O0 = Vector256.Create( (float*)&pack0.Oy );
            // O1 = Vector256.Create( (float*)&pack1.Oy );
            // D0 = Vector256.Create( (float*)&pack0.DInvy );
            // D1 = Vector256.Create( (float*)&pack1.DInvy );
            // t0     = Subtract( Bmin, O0 );
            // t1     = Subtract( Bmax, O0 );
            // t2     = Subtract( Bmin, O1 );
            // t3     = Subtract( Bmax, O1 );
            // t0     = Multiply( t0, D0 );
            // t1     = Multiply( t1, D0 );
            // t2     = Multiply( t2, D1 );
            // t3     = Multiply( t3, D1 );
            // tmin0  = Max( tmin0, t0 );
            // tmax0  = Min( tmax0, t1 );
            // tmin1  = Max( tmin1, t2 );
            // tmax1  = Min( tmax1, t3 );
            //
            // O0 = Vector256.Create( (float*)&pack0.Oz );
            // O1 = Vector256.Create( (float*)&pack1.Oz );
            // D0 = Vector256.Create( (float*)&pack0.DInvz );
            // D1 = Vector256.Create( (float*)&pack1.DInvz );
            // Bmin   = BroadcastScalarToVector256( pAABB+2 );
            // Bmax   = BroadcastScalarToVector256( pAABB+5 ); 
            // t0     = Subtract( Bmin, O0 );
            // t1     = Subtract( Bmax, O0 );
            // t2     = Subtract( Bmin, O1 );
            // t3     = Subtract( Bmax, O1 );
            // t0     = Multiply( t0, D0 );
            // t1     = Multiply( t1, D0 );
            // t2     = Multiply( t2, D1 );
            // t3     = Multiply( t3, D1 );
            // tmin0  = Max( tmin0, t0 );
            // tmax0  = Min( tmax0, t1 );
            // tmin1  = Max( tmin1, t2 );
            // tmax1  = Min( tmax1, t3 );

            ///////////////////////////////////////////////////////////////////////////////////
            // VERSION 2:  same thing, but using fmsub   
            //  MSVC likes to spill unless you spell things out for out very explicitly...
            ///////////////////////////////////////////////////////////////////////////////////
            // Vector256<float> ONE =  _mm256_broadcastss_ps( _mm_set_ss(1.0f) );
            // Vector256<float> Bmin  = BroadcastScalarToVector256( pAABB+0 );
            // Vector256<float> Bmax  = BroadcastScalarToVector256( pAABB+3 );
            //  Vector256<float> O0 = Vector256.Create( (float*)&pack0.Ox );
            //  Vector256<float> O1 = Vector256.Create( (float*)&pack1.Ox );
            //  Vector256<float> D0 = Vector256.Create( (float*)&pack0.DInvx );
            //  Vector256<float> D1 = Vector256.Create( (float*)&pack1.DInvx );
            // Vector256<float> t0 =  MultiplySubtract( Bmin,ONE, O0 );
            // Vector256<float> t1 =  MultiplySubtract( Bmax,ONE, O0 );
            // Vector256<float> t2 =  MultiplySubtract( Bmin,ONE, O1 );
            // Vector256<float> t3 =  MultiplySubtract( Bmax,ONE, O1 );
            // Vector256<float> tmin0 = Multiply(t0,D0 );
            // Vector256<float> tmax0 = Multiply(t1,D0 );
            // Vector256<float> tmin1 = Multiply(t2,D1 );
            // Vector256<float> tmax1 = Multiply(t3,D1 );
            // Bmin   = BroadcastScalarToVector256( pAABB+1 );
            // Bmax   = BroadcastScalarToVector256( pAABB+4 );
            // 
            // O0 = Vector256.Create( (float*)&pack0.Oy );
            // O1 = Vector256.Create( (float*)&pack1.Oy );
            // D0 = Vector256.Create( (float*)&pack0.DInvy );
            // D1 = Vector256.Create( (float*)&pack1.DInvy );
            // t0     = MultiplySubtract( Bmin,ONE, O0 );
            // t1     = MultiplySubtract( Bmax,ONE, O0 );
            // t2     = MultiplySubtract( Bmin,ONE, O1 );
            // t3     = MultiplySubtract( Bmax,ONE, O1 );
            // t0     = Multiply( t0, D0 );
            // t1     = Multiply( t1, D0 );
            // t2     = Multiply( t2, D1 );
            // t3     = Multiply( t3, D1 );
            // tmin0  = Max( tmin0, t0 );
            // tmax0  = Min( tmax0, t1 );
            // tmin1  = Max( tmin1, t2 );
            // tmax1  = Min( tmax1, t3 );
            // 
            // O0 = Vector256.Create( (float*)&pack0.Oz );
            // O1 = Vector256.Create( (float*)&pack1.Oz );
            // D0 = Vector256.Create( (float*)&pack0.DInvz );
            // D1 = Vector256.Create( (float*)&pack1.DInvz );
            // Bmin   = BroadcastScalarToVector256( pAABB+2 );
            // Bmax   = BroadcastScalarToVector256( pAABB+5 ); 
            // t0     = MultiplySubtract( Bmin,ONE, O0 );
            // t1     = MultiplySubtract( Bmax,ONE, O0 );
            // t2     = MultiplySubtract( Bmin,ONE, O1 );
            // t3     = MultiplySubtract( Bmax,ONE, O1 );
            // t0     = Multiply( t0, D0 );
            // t1     = Multiply( t1, D0 );
            // t2     = Multiply( t2, D1 );
            // t3     = Multiply( t3, D1 );
            // tmin0  = Max( tmin0, t0 );
            // tmax0  = Min( tmax0, t1 );
            // tmin1  = Max( tmin1, t2 );
            // tmax1  = Min( tmax1, t3 );

            ///////////////////////////////////////////////////////////////////////////////////
            // VERSION 3:  two muls, then fmsubs  
            //  MSVC likes to spill unless you spell things out for out very explicitly...
            ///////////////////////////////////////////////////////////////////////////////////

            var Bmin = BroadcastScalarToVector256(pAABB + 0);
            var Bmax = BroadcastScalarToVector256(pAABB + 3);
            var O0 = pack0.Ox;// Vector256.Create((float*)&pack0.Ox);
            var O1 = pack1.Ox;// Vector256.Create((float*)&pack1.Ox);
            var D0 = pack0.DInvx;// Vector256.Create((float*)&pack0.DInvx);
            var D1 = pack1.DInvx;// Vector256.Create((float*)&pack1.DInvx);
            var r0 = Multiply(O0, D0);
            var r1 = Multiply(O1, D1);
            var tmin0 = MultiplySubtract(Bmin, D0, r0);
            var tmax0 = MultiplySubtract(Bmax, D0, r0);
            var tmin1 = MultiplySubtract(Bmin, D1, r1);
            var tmax1 = MultiplySubtract(Bmax, D1, r1);

            Bmin = BroadcastScalarToVector256(pAABB + 1);
            Bmax = BroadcastScalarToVector256(pAABB + 4);
            O0 = pack0.Ox;// Vector256.Create((float*)&pack0.Oy);
            O1 = pack1.Ox;// Vector256.Create((float*)&pack1.Oy);
            D0 = pack0.DInvx;// Vector256.Create((float*)&pack0.DInvy);
            D1 = pack1.DInvx;// Vector256.Create((float*)&pack1.DInvy);
            r0 = Multiply(O0, D0);
            r1 = Multiply(O1, D1);
            Vector256<float> t0 = MultiplySubtract(Bmin, D0, r0);
            Vector256<float> t1 = MultiplySubtract(Bmax, D0, r0);
            Vector256<float> t2 = MultiplySubtract(Bmin, D1, r1);
            Vector256<float> t3 = MultiplySubtract(Bmax, D1, r1);
            tmin0 = Max(tmin0, t0);
            tmax0 = Min(tmax0, t1);
            tmin1 = Max(tmin1, t2);
            tmax1 = Min(tmax1, t3);

            Bmin = BroadcastScalarToVector256(pAABB + 2);
            Bmax = BroadcastScalarToVector256(pAABB + 5);
            O0 = pack0.Ox;// Vector256.Create((float*)&pack0.Oz);
            O1 = pack1.Ox;// Vector256.Create((float*)&pack1.Oz);
            D0 = pack0.DInvx;// Vector256.Create((float*)&pack0.DInvz);
            D1 = pack1.DInvx;// Vector256.Create((float*)&pack1.DInvz);
            r0 = Multiply(O0, D0);
            r1 = Multiply(O1, D1);
            t0 = MultiplySubtract(Bmin, D0, r0);
            t1 = MultiplySubtract(Bmax, D0, r0);
            t2 = MultiplySubtract(Bmin, D1, r1);
            t3 = MultiplySubtract(Bmax, D1, r1);
            tmin0 = Max(tmin0, t0);
            tmax0 = Min(tmax0, t1);
            tmin1 = Max(tmin1, t2);
            tmax1 = Min(tmax1, t3);


            // andnot -> uses sign-bit trick for tmax>=0
            Vector256<float> l0 = Min(tmax0, pack0.TMax);
            Vector256<float> l1 = Min(tmax1, pack1.TMax);
            Vector256<float> hit0 = AndNot(tmax0, CompareLessThanOrEqual(tmin0, l0));
            Vector256<float> hit1 = AndNot(tmax1, CompareLessThanOrEqual(tmin1, l1));

            long mask0 = (long)MoveMask(hit0);
            long mask1 = (long)MoveMask(hit1);
            long nMergedMask = (mask1 << 8) | mask0;

            *((ushort*)(pMasks + g)) = (ushort)nMergedMask;
            nHitPopulation += (long)Popcnt.X64.PopCount((ulong)nMergedMask);
        }

        for (; g < nGroups; g++)
        {
            ref RayPacket pack = ref *pPackets[g];
            Vector256<float> Bmin0 = BroadcastScalarToVector256(pAABB + 0);
            Vector256<float> Bmax0 = BroadcastScalarToVector256(pAABB + 3);
            Vector256<float> tmin0 = Multiply(Subtract(Bmin0, pack.Ox), pack.DInvx);
            Vector256<float> tmax0 = Multiply(Subtract(Bmax0, pack.Ox), pack.DInvx);

            Bmin0 = BroadcastScalarToVector256(pAABB + 1);
            Bmax0 = BroadcastScalarToVector256(pAABB + 4);
            tmin0 = Max(tmin0, Multiply(Subtract(Bmin0, pack.Oy), pack.DInvy));
            tmax0 = Min(tmax0, Multiply(Subtract(Bmax0, pack.Oy), pack.DInvy));

            Bmin0 = BroadcastScalarToVector256(pAABB + 2);
            Bmax0 = BroadcastScalarToVector256(pAABB + 5);
            tmin0 = Max(tmin0, Multiply(Subtract(Bmin0, pack.Oz), pack.DInvz));
            tmax0 = Min(tmax0, Multiply(Subtract(Bmax0, pack.Oz), pack.DInvz));

            Vector256<float> l0 = Min(tmax0, pack.TMax);

            Vector256<float> hit = AndNot(tmax0, CompareLessThanOrEqual(tmin0, l0));

            long mask = (long)MoveMask(hit);
            nHitPopulation += (long)Popcnt.X64.PopCount((ulong)mask);
            pMasks[g] = (byte)mask;
        }

        return nHitPopulation;
    }


    // Thanks to Fabian "Rygorous" Giesen for the idea of using a shuffle table
    //
    static readonly Vector128<byte>[] SHUFFLE_TABLE = {
         Vector128.Create(12,13,14,15, 8, 9,10,11, 4, 5, 6, 7, 0, 1, 2, 3).AsByte(),
         Vector128.Create( 0, 1, 2, 3,12,13,14,15, 8, 9,10,11, 4, 5, 6, 7).AsByte(),
         Vector128.Create( 4, 5, 6, 7,12,13,14,15, 8, 9,10,11, 0, 1, 2, 3).AsByte(),
         Vector128.Create( 0, 1, 2, 3, 4, 5, 6, 7,12,13,14,15, 8, 9,10,11).AsByte(),
         Vector128.Create( 8, 9,10,11,12,13,14,15, 4, 5, 6, 7, 0, 1, 2, 3).AsByte(),
         Vector128.Create( 0, 1, 2, 3, 8, 9,10,11,12,13,14,15, 4, 5, 6, 7).AsByte(),
         Vector128.Create( 4, 5, 6, 7, 8, 9,10,11,12,13,14,15, 0, 1, 2, 3).AsByte(),
         Vector128.Create( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15).AsByte(),

         Vector128.Create(12,13,14,15, 8, 9,10,11, 4, 5, 6, 7, 0, 1, 2, 3).AsByte(),
         Vector128.Create( 0, 1, 2, 3,12,13,14,15, 8, 9,10,11, 4, 5, 6, 7).AsByte(),
         Vector128.Create( 4, 5, 6, 7,12,13,14,15, 8, 9,10,11, 0, 1, 2, 3).AsByte(),
         Vector128.Create( 0, 1, 2, 3, 4, 5, 6, 7,12,13,14,15, 8, 9,10,11).AsByte(),
         Vector128.Create( 8, 9,10,11,12,13,14,15, 4, 5, 6, 7, 0, 1, 2, 3).AsByte(),
         Vector128.Create( 0, 1, 2, 3, 8, 9,10,11,12,13,14,15, 4, 5, 6, 7).AsByte(),
         Vector128.Create( 4, 5, 6, 7, 8, 9,10,11,12,13,14,15, 0, 1, 2, 3).AsByte(),
         Vector128.Create( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15).AsByte(),
    };

    static void ReorderRays(ref StackFrame frame, long nGroups)
    {
        RayPacket** pPackets = (RayPacket**)Unsafe.AsPointer(ref frame.pActivePackets[0]);

        uint* pReorderIDs = stackalloc uint[MAX_TRACER_SIZE];

        long nHits = 0;
        long nFirstMiss = 8 * nGroups;

        byte* pRays = (byte*)frame.pRays;
        for (long i = 0; i < nGroups; i++)
        {
            uint* pPacketRayIDs = pPackets[i]->RayOffsets;

            int hits = (int)frame.pMasks[i];
            ulong hit_lo = (ulong)(hits & 0x0f);
            ulong hit_hi = (ulong)(hits & 0xf0) >> 4;
            ulong pop_lo = Popcnt.X64.PopCount(hit_lo);
            ulong pop_hi = Popcnt.X64.PopCount(hit_hi);

            // load lo/hi ID pairs
            var id_lo = Sse2.LoadAlignedVector128(pPacketRayIDs);
            var id_hi = Sse2.LoadAlignedVector128(pPacketRayIDs + 4);

            // store hit/miss iDs
            var shuf_lo = Sse3.Shuffle(id_lo.AsInt32(), SHUFFLE_TABLE[hit_lo].GetElement(0));
            var shuf_hi = Sse3.Shuffle(id_hi.AsInt32(), SHUFFLE_TABLE[hit_hi].GetElement(0));
            Sse2.StoreAligned(&pReorderIDs[nHits], shuf_lo.AsUInt32());
            nHits += (long)pop_lo;
            Sse2.Store(&pReorderIDs[nHits], shuf_hi.AsUInt32());
            nHits += (long)pop_hi;

            // NOTE: Hits MUST be written before misses, or a full-hit packet can corrupt the miss area
            Sse2.StoreAligned(&pReorderIDs[nFirstMiss - 4], shuf_lo.AsUInt32());
            nFirstMiss -= 4 - (long)pop_lo;
            Sse2.StoreAligned(&pReorderIDs[nFirstMiss - 4], shuf_hi.AsUInt32());
            nFirstMiss -= 4 - (long)pop_hi;
        }

        ReadRaysLoopArgs args;
        args.pPackets = pPackets;
        args.pRayIDs = pReorderIDs;
        args.pRays = (byte*)pRays;
        ReadRaysLoop(ref args, nGroups);
    }

    static void TransposePacket(Vector256<float>* pOut, Vector256<float>* p)
    {
        // Transpose a set of 8 m256's  
        //  a 0 1 2 3 4 5 6 7
        //  b ....
        //  c ....

        //  ===>
        //  a0 b0 c0 d0 e0 f0 g0 h0
        //  a1 b1 c1 d1 e1 f1 g1 h1 
        //  ....
        //
        Vector256<float>* pLower = (Vector256<float>*)p;
        Vector256<float>* pUpper = (Vector256<float>*)(((byte*)p) + 16);

        Vector128<float> LOADPS(void* x) => Sse.LoadAlignedVector128((float*)(x));
        Vector256<float> l0 = Vector256.Create(LOADPS(pLower + 4), LOADPS(pLower + 0)); //0123(a) 0123(e)
        Vector256<float> l1 = Vector256.Create(LOADPS(pLower + 5), LOADPS(pLower + 1)); //0123(b) 0123(f)
        Vector256<float> l2 = Vector256.Create(LOADPS(pLower + 6), LOADPS(pLower + 2)); //0123(c) 0123(g)
        Vector256<float> l3 = Vector256.Create(LOADPS(pLower + 7), LOADPS(pLower + 3)); //0123(d) 0123(h)
        Vector256<float> l4 = Vector256.Create(LOADPS(pUpper + 4), LOADPS(pUpper + 0)); //4567(a) 4567(e)
        Vector256<float> l5 = Vector256.Create(LOADPS(pUpper + 5), LOADPS(pUpper + 1)); //4567(b) 4567(f)
        Vector256<float> l6 = Vector256.Create(LOADPS(pUpper + 6), LOADPS(pUpper + 2)); //4567(c) 4567(g)
        Vector256<float> l7 = Vector256.Create(LOADPS(pUpper + 7), LOADPS(pUpper + 3)); //4567(d) 4567(h)

        Vector256<float> t0 = Shuffle(l0, l1, SHUFFLE(0, 1, 0, 1)); // a0a1 b0b1  e0e1  f0f1
        Vector256<float> t1 = Shuffle(l2, l3, SHUFFLE(0, 1, 0, 1)); // c0c1 d0d1  g0g1  h0h1
        Vector256<float> t2 = UnpackLow(t0, t1);                     // a0c0  a1c1  e0 g0  e1g1
        Vector256<float> t3 = UnpackHigh(t0, t1);                     // a0c0  a1c1  e0 g0  e1g1
        t0 = UnpackLow(t2, t3);                            // a0c0  a1c1  e0 g0  e1g1
        t1 = UnpackHigh(t2, t3);                            // b0d0  b1d1  f0 h0  f1h1
        StoreAligned((float*)(pOut + 0), t0);                     // a0 b0 c0 d0 e0 f0 g0 h0
        StoreAligned((float*)(pOut + 1), t1);                     // a1 b1 c1 d1 e1 f1 g1 h1

        t0 = Shuffle(l0, l1, SHUFFLE(2, 3, 2, 3)); // 2 and 3
        t1 = Shuffle(l2, l3, SHUFFLE(2, 3, 2, 3)); // 
        t2 = UnpackLow(t0, t1);                     // 
        t3 = UnpackHigh(t0, t1);                     // 
        t0 = UnpackLow(t2, t3);                     // 
        t1 = UnpackHigh(t2, t3);                     // 
        StoreAligned((float*)(pOut + 2), t0);              // 
        StoreAligned((float*)(pOut + 3), t1);              // 

        t0 = Shuffle(l4, l5, SHUFFLE(0, 1, 0, 1)); // 4 and 5
        t1 = Shuffle(l6, l7, SHUFFLE(0, 1, 0, 1)); // 
        t2 = UnpackLow(t0, t1);                     // 
        t3 = UnpackHigh(t0, t1);                     // 
        t0 = UnpackLow(t2, t3);                     // 
        t1 = UnpackHigh(t2, t3);                     // 
        StoreAligned((float*)(pOut + 4), t0);              // 
        StoreAligned((float*)(pOut + 5), t1);              // 

        t0 = Shuffle(l4, l5, SHUFFLE(2, 3, 2, 3)); // 6 and 7
        t1 = Shuffle(l6, l7, SHUFFLE(2, 3, 2, 3)); // 
        t2 = UnpackLow(t0, t1);                     // 
        t3 = UnpackHigh(t0, t1);                     // 
        t0 = UnpackLow(t2, t3);                     // 
        t1 = UnpackHigh(t2, t3);                     // 
        StoreAligned((float*)(pOut + 6), t0);              // 
        StoreAligned((float*)(pOut + 7), t1);              // 
    }


    static long RemoveMissedGroups(Span<Pointer<RayPacket>> pGroups, byte* pMasks, int nGroups)
    {
        int nHit = 0;
        while (true)
        {
            // skip in-place hits at beginning
            while (pMasks[nHit] != 0)
            {
                nHit++;
                if (nHit == nGroups)
                    return nGroups;
            }

            // skip in-place misses at end
            long mask;
            do
            {
                --nGroups;
                if (nHit == nGroups)
                    return nGroups;
                mask = pMasks[nGroups];

            } while (mask == 0);

            RayPacket* h = pGroups[nHit];
            RayPacket* m = pGroups[nGroups];
            pGroups[nHit] = m;
            pGroups[nGroups] = h;
            pMasks[nHit] = (byte)mask;
        }
    }


    static void BuildPacketsByOctant(RayPacket* pPackets, ref Tracer pTracer, uint* pOctantPacketCounts)
    {

        byte* pRayOctants = pTracer.pRayOctants;
        ushort* pOctantRayCounts = (ushort*)Unsafe.AsPointer(ref pTracer.pOctantCounts[0]);

        uint nRays = pTracer.nRays;
        Ray* pRays = pTracer.pRays;

        // counts into offsets via prefix sum
        uint* pOctantOffsets = stackalloc uint[8];
        pOctantOffsets[0] = 0;
        for (long i = 1; i < 8; i++)
            pOctantOffsets[i] = pOctantOffsets[i - 1] + pOctantRayCounts[i - 1];

        // bin rays by octant
        ushort* pIDsByOctant = stackalloc ushort[MAX_TRACER_SIZE];
        for (long i = 0; i < nRays; i++)
            pIDsByOctant[pOctantOffsets[pRayOctants[i]]++] = (ushort)i;

        for (long i = 0; i < 8; i++)
        {
            uint nOctantRays = pOctantRayCounts[i];
            if (nOctantRays == 0)
                continue;

            uint offs = pOctantOffsets[i] - pOctantRayCounts[i];
            uint nPacks = RoundUp8(nOctantRays) / 8;

            // build a padded ID list
            uint* IDs = stackalloc uint[MAX_TRACER_SIZE];
            for (uint k = 0; k < nOctantRays; k++)
                IDs[k] = pIDsByOctant[offs + k] * (uint)sizeof(Ray);
            uint last = IDs[nOctantRays - 1];
            while ((nOctantRays & 7) != 0)
                IDs[nOctantRays++] = last;

            for (uint p = 0; p < nPacks; p++)
            {
                RayPacket* pPack = pPackets++;
                ReadRays(pPack, (byte*)pRays, IDs + 8 * p);
            }

            pOctantPacketCounts[i] = nPacks;
        }
    }
}