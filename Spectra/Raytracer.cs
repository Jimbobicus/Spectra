using System;

namespace Spectra;

struct RayHitInfo
{
    public const uint NULL_PRIMID = 0xffffffff;
    public float u;
    public float v;
    public uint nPrimID;
    public float t;
}


interface IAllocator
{
    IntPtr Malloc(long nBytes, long nAlign);
    void Free(IntPtr pBytes);
};


unsafe struct Mesh
{
    public float* pVertexPositions;
    public uint* pIndices;
    public uint nVertexStride;
    public uint nTriangleStride;
    public uint nTriangles;
};

unsafe struct RayData
{
    public fixed float O[3]; public float TMax;
    public fixed float D[3]; float _Pad;
}

interface AcceleratorHandle { }
interface TracerHandle { }

unsafe static class Raytracer
{
    public const int MAX_TRACER_SIZE = 4096;

    public static void Init(IAllocator pAlloc)
    {
        throw new NotImplementedException();
    }
    public static void Shutdown()
    {
        throw new NotImplementedException();
    }

    public static AcceleratorHandle CreateAccelerator(ref Mesh pMesh)
    {
        throw new NotImplementedException();
    }
    public static void ReleaseAccelerator(AcceleratorHandle hAccel)
    {
        throw new NotImplementedException();
    }

    public static TracerHandle CreateTracer(long nMaxRays, AcceleratorHandle hAccel)
    {
        throw new NotImplementedException();
    }
    public static void ReleaseTracer(TracerHandle pStream)
    {
        throw new NotImplementedException();
    }

    public static void ResetTracer(TracerHandle hTracer)
    {
        throw new NotImplementedException();
    }
    public static void AddRay(TracerHandle hTracer, RayData* pRay)
    {
        throw new NotImplementedException();
    }
    public static void ReadRayOrigin(TracerHandle hTracer, float* pOrigin, long nRay)
    {
        throw new NotImplementedException();
    }
    public static void ReadRayDirection(TracerHandle hTracer, float* pDir, long nRay)
    {
        throw new NotImplementedException();
    }
    public static void ReadRayData(TracerHandle hTracer, long nRay, RayData* pRayOut)
    {
        throw new NotImplementedException();
    }
    public static long GetRayCount(TracerHandle hTracer)
    {
        throw new NotImplementedException();
    }

    public static void Trace(TracerHandle hTracer, RayHitInfo* pHitsOut)
    {
        throw new NotImplementedException();
    }
    public static void OcclusionTrace(TracerHandle hTracer, byte* pOcclusionOut)
    {
        throw new NotImplementedException();
    }
}
