using System;
using System.Runtime.InteropServices;

namespace Spectra;

static class Allocator
{
    static IAllocator allocator = new DefaultAllocator();

    public static void InitMalloc(IAllocator? pAlloc)
    {
        if (pAlloc is not null)
            allocator = pAlloc;
    }

    public static IntPtr Malloc(long bytes, long align) => allocator.Malloc(bytes, align);

    public static void Free(IntPtr bytes) => allocator.Free(bytes);

    unsafe class DefaultAllocator : IAllocator
    {
        public IntPtr Malloc(long bytes, long align) => new(NativeMemory.AlignedAlloc((nuint)bytes, (nuint)align));

        public void Free(IntPtr bytes) => NativeMemory.AlignedFree(bytes.ToPointer());
    }
}
