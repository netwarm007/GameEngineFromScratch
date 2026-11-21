#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace My {

enum class RhiBackend { D3d12, OpenGL, Metal, Vulkan };

std::string_view RhiBackendName(RhiBackend backend);

struct RhiStartupOptions {
    std::optional<RhiBackend> requestedBackend;
    bool listBackends = false;
    // Borrowed strings; retain this pointer array for the application's lifetime.
    // Includes argv[0] and a trailing nullptr.
    std::vector<char*> applicationArguments;
};

struct RhiStartupParseResult {
    std::optional<RhiStartupOptions> options;
    std::string error;
};

RhiStartupParseResult ParseRhiStartupArguments(
    std::span<char* const> arguments);

enum class RhiAvailability {
    NotCompiled,
    RuntimeUnavailable,
    ReadyForInitialization
};

struct RhiBackendAvailability {
    RhiBackend backend;
    RhiAvailability state;
    std::string reason;
};

struct RhiStartupSelection {
    std::optional<RhiBackend> backend;
    std::string error;
};

// Availability is preflight evidence, not successful native initialization.
RhiStartupSelection ResolveRhiStartupBackend(
    std::optional<RhiBackend> requestedBackend, RhiBackend defaultBackend,
    std::span<const RhiBackendAvailability> availability);

}  // namespace My
