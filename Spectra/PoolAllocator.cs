using System;

namespace Spectra;

/// Simple block allocator which allocates storage in chunks and never releases any of it
/// This is intended for aggregating large numbers of temporary allocs which are all released at once
/// It is by no means a general purpose allocator.
unsafe class PoolAllocator
{
    public class ScopedFree
    {
        public ScopedFree(PoolAllocator alloc) { m_rAlloc = (alloc); }
        ~ScopedFree() { m_rAlloc.FreeAll(); }

        PoolAllocator m_rAlloc;
    };

    public class ScopedRecycler
    {
        public ScopedRecycler(PoolAllocator alloc) { m_rAlloc = (alloc); }
        ~ScopedRecycler() { m_rAlloc.Recycle(); }

        PoolAllocator m_rAlloc;
    };

    public PoolAllocator() { m_pHead = null; }
    ~PoolAllocator() { FreeAll(); }

    public T* GetA<T>() where T : unmanaged { return (T*)(GetMore(sizeof(T))); }
    public T* GetSome<T>(long n) where T : unmanaged { return (T*)(GetMore(sizeof(T) * n)); }
    public IntPtr GetMore(long nSize) => throw new NotImplementedException();

    public void FreeAll() => throw new NotImplementedException();

    public void Recycle() => throw new NotImplementedException();


    struct AllocHeader
    {
        public AllocHeader* pNext;     ///< Next block in linked list
        public long nCapacity;       ///< Total Size of block (including header)
        public long nOffset;         ///< Offset of next free byte     
    };

    const int ALIGN = 64;       ///< Align all 'GetMore' calls to this size
    const int CHUNKSIZE = 16 * 1024;  ///< Align all mallocs to this size
    static readonly int HEADER_SIZE = (sizeof(AllocHeader) + ALIGN - 1) & ~(ALIGN - 1);

    AllocHeader* m_pHead;
}
