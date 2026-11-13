#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "D3d/D3d12RHI.hpp"
#include "D3d/D3d12ShadowUtils.hpp"
#include "D3d/D3d12Utility.hpp"

using namespace My;

bool TestPointShadowDirections() {
    const Vector3f position{3.0f, -4.0f, 2.0f};
    Matrix4X4f projection;
    BuildPerspectiveFovRHMatrix(projection, PI / 2.0f, 1.0f, 1.0f, 1000.0f);
    Matrix4X4f matrices[6];
    BuildD3d12PointShadowMatrices(matrices, position, projection);

    // Independent D3D cube-face selection: (major axis, sc, tc).
    const Vector3f directions[] = {
        {4, 1, 2}, {-4, 1, 2}, {1, 4, 2},
        {1, -4, 2}, {1, 2, 4}, {1, 2, -4}
    };
    const float sc[] = {-2, 2, 1, 1, 1, -1};
    const float tc[] = {-1, -1, 2, -2, -2, -2};
    for (size_t i = 0; i < 6; ++i) {
        for (float scale : {-1.9f, -1.0f, 0.0f, 1.0f, 1.9f}) {
            Vector4f point{position[0], position[1], position[2], 1.0f};
            for (size_t axis = 0; axis < 3; ++axis) {
                point[axis] += directions[i][axis] *
                    (axis == i / 2 ? 1.0f : scale);
            }
            Transform(point, matrices[i]);
            const float u = (point[0] / point[3] + 1.0f) * 0.5f;
            const float v = (1.0f - point[1] / point[3]) * 0.5f;
            if (std::abs(u - (sc[i] * scale / 4.0f + 1.0f) * 0.5f) > 0.0001f ||
                std::abs(v - (tc[i] * scale / 4.0f + 1.0f) * 0.5f) > 0.0001f ||
                point[3] <= 0.0f || point[2] < 0.0f || point[2] > point[3]) {
                std::cerr << "Point shadow cube face " << i << " has incorrect projection\n";
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (!TestPointShadowDirections()) return 1;
    if (argc != 4) {
        std::cerr << "Expected the compiled BRDF, mipmap and shadow coordinate shader paths\n";
        return 1;
    }

    std::ifstream input(argv[1], std::ios::binary);
    std::vector<char> shaderData((std::istreambuf_iterator<char>(input)),
                                 std::istreambuf_iterator<char>());
    if (shaderData.empty()) {
        std::cerr << "Could not read the BRDF shader\n";
        return 1;
    }

    D3d12RHI rhi;
    rhi.CreateDevice();
    rhi.EnableDebugLayer();
    rhi.CreateCommandQueues();
    rhi.CreateSyncObjects();
    rhi.CreateCommandPools();
    rhi.CreateCommandLists();
    auto* device = rhi.GetDevice();

    D3D12_SHADER_BYTECODE shader{shaderData.data(), shaderData.size()};
    auto* root = rhi.CreateRootSignature(shader);
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineDesc{};
    pipelineDesc.pRootSignature = root;
    pipelineDesc.CS = shader;
    auto* pipeline = rhi.CreateComputePipeline(pipelineDesc);
    auto* descriptors = rhi.CreateDescriptorHeap(1, L"Compute Test");

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 2;
    desc.Height = 2;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R32G32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource* texture = nullptr;
    if (FAILED(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
            nullptr, IID_PPV_ARGS(&texture)))) {
        std::cerr << "Could not create compute test texture\n";
        return 1;
    }

    for (int i = 0; i < 2; ++i) {
        rhi.BeginCompute();
        rhi.SetComputePipelineState(pipeline, root);
        rhi.BindComputeTexture(texture, descriptors);
        rhi.Dispatch(2, 2, 1);
        rhi.EndCompute(texture);
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    UINT rows;
    UINT64 rowSize, readbackSize;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &layout, &rows, &rowSize,
                                  &readbackSize);
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = readbackSize;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* readback = nullptr;
    HRESULT hr = device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(&readback));
    if (FAILED(hr)) {
        std::cerr << "Could not create readback buffer\n";
        return 1;
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    if (FAILED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                               IID_PPV_ARGS(&allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                          allocator, nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
        std::cerr << "Could not create readback commands\n";
        return 1;
    }
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = texture;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = readback;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = layout;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (FAILED(list->Close())) return 1;
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!event || FAILED(queue->Signal(fence, 1)) ||
        FAILED(fence->SetEventOnCompletion(1, event)) ||
        WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0) {
        std::cerr << "Readback synchronization failed\n";
        return 1;
    }
    void* data = nullptr;
    if (FAILED(readback->Map(0, nullptr, &data))) return 1;
    bool valid = true;
    for (UINT y = 0; y < 2; ++y) {
        const auto* row = reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(data) + layout.Offset + y * layout.Footprint.RowPitch);
        for (UINT x = 0; x < 2; ++x) {
            valid = valid && std::isfinite(row[x * 2]) &&
                    std::isfinite(row[x * 2 + 1]) &&
                    (row[x * 2] > 0.0f || row[x * 2 + 1] > 0.0f);
        }
    }
    readback->Unmap(0, nullptr);

    std::ifstream shadowInput(argv[3], std::ios::binary);
    std::vector<char> shadowShaderData((std::istreambuf_iterator<char>(shadowInput)),
                                       std::istreambuf_iterator<char>());
    if (shadowShaderData.empty()) {
        std::cerr << "Could not read the shadow coordinate shader\n";
        return 1;
    }
    D3D12_SHADER_BYTECODE shadowShader{shadowShaderData.data(), shadowShaderData.size()};
    auto* shadowRoot = rhi.CreateRootSignature(shadowShader);
    D3D12_COMPUTE_PIPELINE_STATE_DESC shadowPipelineDesc{};
    shadowPipelineDesc.pRootSignature = shadowRoot;
    shadowPipelineDesc.CS = shadowShader;
    auto* shadowPipeline = rhi.CreateComputePipeline(shadowPipelineDesc);
    rhi.BeginCompute();
    rhi.SetComputePipelineState(shadowPipeline, shadowRoot);
    rhi.BindComputeTexture(texture, descriptors);
    rhi.Dispatch(1, 1, 1);
    rhi.EndCompute(texture);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator, nullptr))) return 1;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (FAILED(list->Close())) return 1;
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence, 2)) ||
        FAILED(fence->SetEventOnCompletion(2, event)) ||
        WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0) {
        std::cerr << "Shadow coordinate readback synchronization failed\n";
        return 1;
    }
    if (FAILED(readback->Map(0, nullptr, &data))) return 1;
    for (UINT y = 0; y < 2; ++y) {
        const auto* row = reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(data) + layout.Offset + y * layout.Footprint.RowPitch);
        for (UINT x = 0; x < 2; ++x) {
            if (std::abs(row[x * 2] - (x == 0 ? 0.2f : 0.8f)) > 0.0001f ||
                std::abs(row[x * 2 + 1] - (y == 0 ? 0.25f : 0.75f)) > 0.0001f) {
                std::cerr << "Shadow coordinate projection is incorrect\n";
                valid = false;
            }
        }
    }
    readback->Unmap(0, nullptr);
    SafeRelease(&shadowPipeline);
    SafeRelease(&shadowRoot);

    auto* depth = rhi.CreateDepthTextureArray(2, 2, 2);
    auto depthDesc = depth->GetDesc();
    valid = valid && depthDesc.DepthOrArraySize == 2 &&
            depthDesc.Format == DXGI_FORMAT_R32_TYPELESS;
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc{};
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.NumDescriptors = 1;
    ID3D12DescriptorHeap* dsvHeap = nullptr;
    if (FAILED(device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&dsvHeap)))) {
        std::cerr << "Could not create depth descriptor heap\n";
        return 1;
    }
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
    dsv.Format = DXGI_FORMAT_D32_FLOAT;
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
    dsv.Texture2DArray.FirstArraySlice = 1;
    dsv.Texture2DArray.ArraySize = 1;
    device->CreateDepthStencilView(depth, &dsv, dsvHeap->GetCPUDescriptorHandleForHeapStart());
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator, nullptr))) return 1;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = depth;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    list->ResourceBarrier(1, &barrier);
    list->ClearDepthStencilView(dsvHeap->GetCPUDescriptorHandleForHeapStart(),
                                D3D12_CLEAR_FLAG_DEPTH, 0.25f, 0, 0, nullptr);
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    device->GetCopyableFootprints(&depthDesc, 1, 1, 0, &layout, &rows, &rowSize,
                                  &readbackSize);
    src.pResource = depth;
    src.SubresourceIndex = 1;
    dst.PlacedFootprint = layout;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (FAILED(list->Close())) return 1;
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence, 3)) ||
        FAILED(fence->SetEventOnCompletion(3, event)) ||
        WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0) {
        std::cerr << "Depth readback synchronization failed\n";
        return 1;
    }
    if (FAILED(readback->Map(0, nullptr, &data))) return 1;
    for (UINT y = 0; y < 2; ++y) {
        const auto* row = reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(data) + layout.Offset + y * layout.Footprint.RowPitch);
        for (UINT x = 0; x < 2; ++x) valid = valid && std::abs(row[x] - 0.25f) < 0.0001f;
    }
    readback->Unmap(0, nullptr);
    SafeRelease(&dsvHeap);
    SafeRelease(&depth);

    Image rgb;
    rgb.Width = rgb.Height = 2;
    rgb.pitch = 8;
    rgb.data_size = rgb.pitch * rgb.Height;
    rgb.pixel_format = PIXEL_FORMAT::RGB8;
    rgb.data = new uint8_t[rgb.data_size]{
        1, 2, 3, 4, 5, 6, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0};
    auto* rgbTexture = rhi.CreateTextureImage(rgb);
    const auto rgbDesc = rgbTexture->GetDesc();
    valid = valid && rgbDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM;
    device->GetCopyableFootprints(&rgbDesc, 0, 1, 0, &layout, &rows, &rowSize,
                                  &readbackSize);
    for (UINT pass = 0; pass < 2; ++pass) {
        if (pass) {
            rgb.data[0] = 42;
            rhi.UpdateTexture(rgbTexture, rgb);
        }
        if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator, nullptr))) return 1;
        src.pResource = rgbTexture;
        src.SubresourceIndex = 0;
        dst.PlacedFootprint = layout;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        if (FAILED(list->Close())) return 1;
        queue->ExecuteCommandLists(1, lists);
        const UINT64 fenceValue = 4 + pass;
        if (FAILED(queue->Signal(fence, fenceValue)) ||
            FAILED(fence->SetEventOnCompletion(fenceValue, event)) ||
            WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0) {
            std::cerr << "RGB texture readback synchronization failed\n";
            return 1;
        }
        if (FAILED(readback->Map(0, nullptr, &data))) return 1;
        for (UINT y = 0; y < 2; ++y) {
            const auto* row = static_cast<const uint8_t*>(data) +
                              layout.Offset + y * layout.Footprint.RowPitch;
            for (UINT x = 0; x < 2; ++x) {
                for (UINT c = 0; c < 3; ++c)
                    valid = valid && row[x * 4 + c] == rgb.data[y * rgb.pitch + x * 3 + c];
                valid = valid && row[x * 4 + 3] == 255;
            }
        }
        readback->Unmap(0, nullptr);
    }
    std::ifstream mipInput(argv[2], std::ios::binary);
    std::vector<char> mipShaderData((std::istreambuf_iterator<char>(mipInput)),
                                    std::istreambuf_iterator<char>());
    if (mipShaderData.empty()) {
        std::cerr << "Could not read the mipmap shader\n";
        return 1;
    }
    D3D12_SHADER_BYTECODE mipShader{mipShaderData.data(), mipShaderData.size()};
    auto* mipRoot = rhi.CreateRootSignature(mipShader);
    D3D12_COMPUTE_PIPELINE_STATE_DESC mipPipelineDesc{};
    mipPipelineDesc.pRootSignature = mipRoot;
    mipPipelineDesc.CS = mipShader;
    auto* mipPipeline = rhi.CreateComputePipeline(mipPipelineDesc);
    auto* mipTexture = rhi.GenerateTextureMipmaps(rgbTexture, mipPipeline, mipRoot);
    const auto mipDesc = mipTexture->GetDesc();
    valid = valid && mipDesc.MipLevels == 2;
    device->GetCopyableFootprints(&mipDesc, 1, 1, 0, &layout, &rows, &rowSize,
                                  &readbackSize);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator, nullptr))) return 1;
    src.pResource = mipTexture;
    src.SubresourceIndex = 1;
    dst.PlacedFootprint = layout;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (FAILED(list->Close())) return 1;
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence, 6)) ||
        FAILED(fence->SetEventOnCompletion(6, event)) ||
        WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0) {
        std::cerr << "Mipmap readback synchronization failed\n";
        return 1;
    }
    if (FAILED(readback->Map(0, nullptr, &data))) return 1;
    const auto* pixel = static_cast<const uint8_t*>(data) + layout.Offset;
    const uint8_t expected[] = {16, 7, 8, 255};
    for (UINT c = 0; c < 4; ++c) {
        valid = valid && std::abs(static_cast<int>(pixel[c]) - expected[c]) <= 1;
    }
    readback->Unmap(0, nullptr);
    SafeRelease(&mipTexture);

    auto verifyLastMip = [&](Image& image, UINT64 fenceValue,
                             const uint8_t expectedPixel[4]) {
        auto* inputTexture = rhi.CreateTextureImage(image);
        auto* generated = rhi.GenerateTextureMipmaps(inputTexture, mipPipeline, mipRoot);
        const auto generatedDesc = generated->GetDesc();
        device->GetCopyableFootprints(&generatedDesc, generatedDesc.MipLevels - 1, 1, 0,
                                      &layout, &rows, &rowSize, &readbackSize);
        if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator, nullptr))) return false;
        src.pResource = generated;
        src.SubresourceIndex = generatedDesc.MipLevels - 1;
        dst.PlacedFootprint = layout;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        if (FAILED(list->Close())) return false;
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(queue->Signal(fence, fenceValue)) ||
            FAILED(fence->SetEventOnCompletion(fenceValue, event)) ||
            WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0) return false;
        if (FAILED(readback->Map(0, nullptr, &data))) return false;
        const auto* actual = static_cast<const uint8_t*>(data) + layout.Offset;
        bool matches = true;
        for (UINT c = 0; c < 4; ++c)
            matches = matches && std::abs(static_cast<int>(actual[c]) - expectedPixel[c]) <= 1;
        readback->Unmap(0, nullptr);
        SafeRelease(&generated);
        SafeRelease(&inputTexture);
        return matches;
    };

    Image odd;
    odd.Width = 3;
    odd.Height = 1;
    odd.pitch = odd.data_size = 9;
    odd.pixel_format = PIXEL_FORMAT::RGB8;
    odd.data = new uint8_t[9]{0, 0, 0, 0, 0, 0, 255, 0, 0};
    const uint8_t oddExpected[] = {85, 0, 0, 255};
    if (!verifyLastMip(odd, 7, oddExpected)) {
        std::cerr << "Odd-sized mipmap readback failed\n";
        valid = false;
    }

    Image rgb16;
    rgb16.Width = 3;
    rgb16.Height = 1;
    rgb16.pitch = rgb16.data_size = 18;
    rgb16.pixel_format = PIXEL_FORMAT::RGB16;
    rgb16.data = new uint8_t[18]{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0};
    if (!verifyLastMip(rgb16, 8, oddExpected)) {
        std::cerr << "16-bit RGB mipmap readback failed\n";
        valid = false;
    }

    Image bc1;
    bc1.Width = bc1.Height = 4;
    bc1.pitch = bc1.data_size = 8;
    bc1.compressed = true;
    bc1.compress_format = COMPRESSED_FORMAT::BC1;
    bc1.data = new uint8_t[8]{0x00, 0xf8, 0x00, 0xf8, 0, 0, 0, 0};
    const uint8_t bcExpected[] = {255, 0, 0, 255};
    if (!verifyLastMip(bc1, 9, bcExpected)) {
        std::cerr << "Compressed mipmap readback failed\n";
        valid = false;
    }

    SafeRelease(&mipPipeline);
    SafeRelease(&mipRoot);
    SafeRelease(&rgbTexture);

    CloseHandle(event);
    SafeRelease(&fence);
    SafeRelease(&list);
    SafeRelease(&allocator);
    SafeRelease(&queue);
    SafeRelease(&readback);
    SafeRelease(&texture);
    SafeRelease(&descriptors);
    SafeRelease(&pipeline);
    SafeRelease(&root);
    rhi.DestroyAll();
    if (!valid) std::cerr << "GPU readback output was invalid\n";
    return valid ? 0 : 1;
}
