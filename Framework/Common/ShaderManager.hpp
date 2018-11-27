#pragma once
#include <unordered_map>
#include "IShaderManager.hpp"

namespace std
{
	// Specialization for std::hash<Guid> -- this implementation
	// uses std::hash<std::string> on the stringification of the guid
	// to calculate the hash
	template <>
	struct hash<const My::DefaultShaderIndex>
	{
		typedef My::DefaultShaderIndex argument_type;
		typedef std::size_t result_type;

		result_type operator()(argument_type const &index) const
		{
			std::hash<std::int32_t> hasher;
			return static_cast<result_type>(hasher((int32_t)index));
		}
	};
}

namespace My {
    class ShaderManager : implements IShaderManager
    {
    public:
        ShaderManager() = default;
        ~ShaderManager() = default;

        virtual int32_t GetDefaultShaderProgram(DefaultShaderIndex index) final
        {
            return m_DefaultShaders[index];
        }

    protected:
        std::unordered_map<const DefaultShaderIndex, int32_t> m_DefaultShaders;
    };
}