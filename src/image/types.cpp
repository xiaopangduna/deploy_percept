#include "deploy_percept/types.hpp"

namespace deploy_percept
{
namespace image
{

int channels(PixelFormat format)
{
    switch (format)
    {
    case PixelFormat::BGR888:
    case PixelFormat::RGB888:
        return 3;
    case PixelFormat::GRAY8:
        return 1;
    }
    return 0;
}

int ImageView::channels() const
{
    return image::channels(format);
}

std::size_t ImageView::row_bytes() const
{
    if (width <= 0)
    {
        return 0;
    }
    if (stride > 0)
    {
        return static_cast<std::size_t>(stride);
    }
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(channels());
}

std::size_t ImageView::byte_size() const
{
    if (height <= 0)
    {
        return 0;
    }
    return row_bytes() * static_cast<std::size_t>(height);
}

bool ImageView::empty() const
{
    return data == nullptr || width <= 0 || height <= 0;
}

ImageView ImageMut::view() const
{
    ImageView out;
    out.data = data;
    out.width = width;
    out.height = height;
    out.stride = stride;
    out.format = format;
    return out;
}

int ImageMut::channels() const
{
    return view().channels();
}

std::size_t ImageMut::row_bytes() const
{
    return view().row_bytes();
}

std::size_t ImageMut::byte_size() const
{
    return view().byte_size();
}

bool ImageMut::empty() const
{
    return view().empty();
}

} // namespace image
} // namespace deploy_percept
