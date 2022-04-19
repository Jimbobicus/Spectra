using System;
using System.Collections.Generic;
using System.Numerics;

namespace Spectra;

struct AxisAlignedBox
{

    /// Computes the bounding box of a set of points
    public AxisAlignedBox(IEnumerable<Vector3> pPoints)
    {
        m_min = default;
        m_max = default;
        ComputeFromPoints(pPoints);
    }

    /// Constructs the box given two endpoints
    public AxisAlignedBox(Vector3 rMin, Vector3 rMax) { m_min = (rMin); m_max = (rMax); }

    public static bool operator ==(AxisAlignedBox l, AxisAlignedBox r) => l.m_min == r.m_min && l.m_max == r.m_max;
    public static bool operator !=(AxisAlignedBox l, AxisAlignedBox r) => !(l == r);

    /// Returns a reference to the minimum point

    /// Returns a reference to the minimum point.  The caller may modify it
    public Vector3 Min
    {
        get => m_min;
        set => m_min = value;
    }

    /// Returns a reference to the maximum point.  The caller may modify it
    public Vector3 Max
    {
        get => m_max;
        set => m_max = value;
    }

    /// Computes the center of the box
    public Vector3 Center => (m_min + m_max) * 0.5f;

    /// Makes this box equal to the AABB of a set of points
    public void ComputeFromPoints(IEnumerable<Vector3> pPoints)
    {
        foreach (var pt in pPoints)
            Expand(pt);
    }

    /// Expands this box to include the specified point
    public void Expand(Vector3 rPoint)
    {
        m_min.X = MathF.Min(m_min.X, rPoint.X);
        m_min.Y = MathF.Min(m_min.Y, rPoint.Y);
        m_min.Z = MathF.Min(m_min.Z, rPoint.Z);

        m_max.X = MathF.Max(m_max.X, rPoint.X);
        m_max.Y = MathF.Max(m_max.Y, rPoint.Y);
        m_max.Z = MathF.Max(m_max.Z, rPoint.Z);
    }

    /// Expands this box to include the specified box
    public void Merge(AxisAlignedBox rBox)
    {
        m_min.X = MathF.Min(m_min.X, rBox.Min.X);
        m_min.Y = MathF.Min(m_min.Y, rBox.Min.Y);
        m_min.Z = MathF.Min(m_min.Z, rBox.Min.Z);
        m_max.X = MathF.Max(m_max.X, rBox.Max.X);
        m_max.Y = MathF.Max(m_max.Y, rBox.Max.Y);
        m_max.Z = MathF.Max(m_max.Z, rBox.Max.Z);
    }

    /// Checks whether the argument box is fully contained in the calling box
    public bool Contains(AxisAlignedBox rBox)
    {
        if (m_min.X > rBox.m_min.X) return false;
        if (m_min.Y > rBox.m_min.Y) return false;
        if (m_min.Z > rBox.m_min.Z) return false;
        if (m_max.X < rBox.m_max.X) return false;
        if (m_max.Y < rBox.m_max.Y) return false;
        if (m_max.Z < rBox.m_max.Z) return false;
        return true;
    }

    /// Tests whether a point is contained in the box (inclusive.  Points on the edges are counted)
    public bool Contains(Vector3 P)
    {
        if (P.X < m_min.X) return false;
        if (P.Y < m_min.Y) return false;
        if (P.Z < m_min.Z) return false;
        if (P.X > m_max.X) return false;
        if (P.Y > m_max.Y) return false;
        if (P.Z > m_max.Z) return false;
        return true;
    }

    /// Checks whether the intersection of two boxes is non-empty
    public bool Intersects(AxisAlignedBox rBox)
    {
        if (m_min.X > rBox.m_max.X) return false;
        if (m_min.Y > rBox.m_max.Y) return false;
        if (m_min.Z > rBox.m_max.Z) return false;
        if (m_max.X < rBox.m_min.X) return false;
        if (m_max.Y < rBox.m_min.Y) return false;
        if (m_max.Z < rBox.m_min.Z) return false;
        return true;
    }

    /// Cuts an AABB using an axis-aligned split plane
    public void Cut(uint nAxis, float fLocation, AxisAlignedBox rLeft, AxisAlignedBox rRight)
    {
        rLeft.Min = m_min;
        rRight.Max = m_max;

        switch (nAxis)
        {
            case 0:
                rLeft.Max = new(fLocation, m_max.Y, m_max.Z);
                rRight.Min = new(fLocation, m_min.Y, m_min.Z);
                break;

            case 1:
                rLeft.Max = new(m_max.X, fLocation, m_max.Z);
                rRight.Min = new(m_min.X, fLocation, m_min.Z);
                break;
            case 2:
                rLeft.Max = new(m_max.X, m_max.Y, fLocation);
                rRight.Min = new(m_min.X, m_min.Y, fLocation);
                break;
        }
    }

    /// Cuts an AABB using an axis-aligned split plane, returning the lower half
    public void CutLeft(uint nAxis, float fLocation, AxisAlignedBox rLeft)
    {
        rLeft.Min = m_min;

        switch (nAxis)
        {
            case 0:
                rLeft.Max = new(fLocation, m_max.Y, m_max.Z);
                break;

            case 1:
                rLeft.Max = new(m_max.X, fLocation, m_max.Z);
                break;
            case 2:
                rLeft.Max = new(m_max.X, m_max.Y, fLocation);
                break;
        }
    }

    /// Cuts an AABB using an axis-aligned split plane, returning the upper half
    public void CutRight(uint nAxis, float fLocation, AxisAlignedBox rRight)
    {
        rRight.Max = m_max;

        switch (nAxis)
        {
            case 0:
                rRight.Min = new(fLocation, m_min.Y, m_min.Z);
                break;

            case 1:
                rRight.Min = new(m_min.X, fLocation, m_min.Z);
                break;
            case 2:
                rRight.Min = new(m_min.X, m_min.Y, fLocation);
                break;
        }
    }

    /// Sets this box to the intersection of this box with another
    public void Intersect(AxisAlignedBox rBox)
    {
        m_min = Vector3.Max(m_min, rBox.Min);
        m_max = Vector3.Min(m_max, rBox.Max);
    }

    /// Tests whether the box's Min point is <= its Max point
    public bool IsValid() => m_min.X <= m_max.X && m_min.Y <= m_max.Y && m_min.Z <= m_max.Z;

    Vector3 m_min;
    Vector3 m_max;
}
