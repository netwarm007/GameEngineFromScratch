#include <iostream>
#include <fstream>
#include "OpenGLESGraphicsManager.hpp"
#include "AssetLoader.hpp"
#include "IApplication.hpp"
#include "SceneManager.hpp"
#include "IPhysicsManager.hpp"

using namespace My;
using namespace std;

const char VS_SHADER_SOURCE_FILE[] = "Shaders/basic_vs.glsl";
const char PS_SHADER_SOURCE_FILE[] = "Shaders/basic_ps.glsl";
#ifdef DEBUG
const char DEBUG_VS_SHADER_SOURCE_FILE[] = "Shaders/debug_vs.glsl";
const char DEBUG_PS_SHADER_SOURCE_FILE[] = "Shaders/debug_ps.glsl";
#endif

namespace My {
    extern AssetLoader* g_pAssetLoader;

    static void OutputShaderErrorMessage(unsigned int shaderId, const char* shaderFilename)
    {
        int logSize, i;
        char* infoLog;
        ofstream fout;

        // Get the size of the string containing the information log for the failed shader compilation message.
        glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &logSize);

        // Increment the size by one to handle also the null terminator.
        logSize++;

        // Create a char buffer to hold the info log.
        infoLog = new char[logSize];
        if(!infoLog)
        {
                return;
        }

        // Now retrieve the info log.
        glGetShaderInfoLog(shaderId, logSize, NULL, infoLog);

        // Open a file to write the error message to.
        fout.open("shader-error.txt");

        // Write out the error message.
        for(i=0; i<logSize; i++)
        {
                fout << infoLog[i];
        }

        // Close the file.
        fout.close();

        // Pop a message up on the screen to notify the user to check the text file for compile errors.
        cerr << "Error compiling shader.  Check shader-error.txt for message." << shaderFilename << endl;

        return;
    }

    static void OutputLinkerErrorMessage(unsigned int programId)
    {
        int logSize, i;
        char* infoLog;
        ofstream fout;


        // Get the size of the string containing the information log for the failed shader compilation message.
        glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &logSize);

        // Increment the size by one to handle also the null terminator.
        logSize++;

        // Create a char buffer to hold the info log.
        infoLog = new char[logSize];
        if(!infoLog)
        {
                return;
        }

        // Now retrieve the info log.
        glGetProgramInfoLog(programId, logSize, NULL, infoLog);

        // Open a file to write the error message to.
        fout.open("linker-error.txt");

        // Write out the error message.
        for(i=0; i<logSize; i++)
        {
                fout << infoLog[i];
        }

        // Close the file.
        fout.close();

        // Pop a message up on the screen to notify the user to check the text file for linker errors.
        cerr << "Error compiling linker.  Check linker-error.txt for message." << endl;
    }
}

int OpenGLESGraphicsManager::Initialize()
{
    int result;

    result = GraphicsManager::Initialize();

    if (result) {
        return result;
    }

    auto opengl_info = {GL_VENDOR, GL_RENDERER, GL_VERSION, GL_EXTENSIONS};
    for (auto name : opengl_info) {
        auto info = glGetString(name);
        printf("OpenGL Info: %s", info);
    }

	// Set the depth buffer to be entirely cleared to 1.0 values.
	glClearDepthf(1.0f);

	// Enable depth testing.
	glEnable(GL_DEPTH_TEST);

	// Set the polygon winding to front facing for the right handed system.
	glFrontFace(GL_CCW);

	// Enable back face culling.
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

    return result;
}

void OpenGLESGraphicsManager::Finalize()
{
    GraphicsManager::Finalize();
}

void OpenGLESGraphicsManager::Clear()
{
    GraphicsManager::Clear();

    // Set the color to clear the screen to.
    glClearColor(0.2f, 0.3f, 0.4f, 1.0f);
    // Clear the screen and depth buffer.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void OpenGLESGraphicsManager::Draw()
{
    GraphicsManager::Draw();

    glFlush();
}

bool OpenGLESGraphicsManager::SetPerFrameShaderParameters(GLuint shader)
{
    unsigned int location;

    // Set the world matrix in the vertex shader.
    location = glGetUniformLocation(shader, "worldMatrix");
    if(location == -1)
    {
            return false;
    }
    glUniformMatrix4fv(location, 1, false, m_DrawFrameContext.m_worldMatrix);

    // Set the view matrix in the vertex shader.
    location = glGetUniformLocation(shader, "viewMatrix");
    if(location == -1)
    {
            return false;
    }
    glUniformMatrix4fv(location, 1, false, m_DrawFrameContext.m_viewMatrix);

    // Set the projection matrix in the vertex shader.
    location = glGetUniformLocation(shader, "projectionMatrix");
    if(location == -1)
    {
            return false;
    }
    glUniformMatrix4fv(location, 1, false, m_DrawFrameContext.m_projectionMatrix);

    // Set lighting parameters for PS shader
    location = glGetUniformLocation(shader, "lightPosition");
    if(location == -1)
    {
            return false;
    }
    glUniform3fv(location, 1, m_DrawFrameContext.m_lightPosition);

    location = glGetUniformLocation(shader, "lightColor");
    if(location == -1)
    {
            return false;
    }
    glUniform4fv(location, 1, m_DrawFrameContext.m_lightColor);

    location = glGetUniformLocation(shader, "ambientColor");
    if(location == -1)
    {
            return false;
    }
    glUniform4fv(location, 1, m_DrawFrameContext.m_ambientColor);
    return true;
}

bool OpenGLESGraphicsManager::SetPerBatchShaderParameters(GLuint shader, const char* paramName, const Matrix4X4f& param)
{
    unsigned int location;

    location = glGetUniformLocation(shader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniformMatrix4fv(location, 1, false, param);

    return true;
}

bool OpenGLESGraphicsManager::SetPerBatchShaderParameters(GLuint shader, const char* paramName, const Vector3f& param)
{
    unsigned int location;

    location = glGetUniformLocation(shader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniform3fv(location, 1, param);

    return true;
}

bool OpenGLESGraphicsManager::SetPerBatchShaderParameters(GLuint shader, const char* paramName, const float param)
{
    unsigned int location;

    location = glGetUniformLocation(shader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniform1f(location, param);

    return true;
}

bool OpenGLESGraphicsManager::SetPerBatchShaderParameters(GLuint shader, const char* paramName, const int param)
{
    unsigned int location;

    location = glGetUniformLocation(shader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniform1i(location, param);

	return true;
}

bool OpenGLESGraphicsManager::SetPerBatchShaderParameters(GLuint shader, const char* paramName, const bool param)
{
    unsigned int location;

    location = glGetUniformLocation(shader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniform1f(location, param);

	return true;
}

void OpenGLESGraphicsManager::InitializeBuffers(const Scene& scene)
{
    // Geometries
    for (auto _it : scene.GeometryNodes)
    {
        auto pGeometryNode = _it.second.lock();
        if (pGeometryNode && pGeometryNode->Visible()) 
        {
            auto pGeometry = scene.GetGeometry(pGeometryNode->GetSceneObjectRef());
            assert(pGeometry);
            auto pMesh = pGeometry->GetMesh().lock();
            if (!pMesh) continue;

            // Set the number of vertex properties.
            auto vertexPropertiesCount = pMesh->GetVertexPropertiesCount();

            // Allocate an OpenGL vertex array object.
            GLuint vao;
            glGenVertexArrays(1, &vao);

            // Bind the vertex array object to store all the buffers and vertex attributes we create here.
            glBindVertexArray(vao);

            GLuint buffer_id;

            for (uint32_t i = 0; i < vertexPropertiesCount; i++)
            {
                const SceneObjectVertexArray& v_property_array = pMesh->GetVertexPropertyArray(i);
                auto v_property_array_data_size = v_property_array.GetDataSize();
                auto v_property_array_data = v_property_array.GetData();

                // Generate an ID for the vertex buffer.
                glGenBuffers(1, &buffer_id);

                // Bind the vertex buffer and load the vertex (position and color) data into the vertex buffer.
                glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
                glBufferData(GL_ARRAY_BUFFER, v_property_array_data_size, v_property_array_data, GL_STATIC_DRAW);

                glEnableVertexAttribArray(i);

                switch (v_property_array.GetDataType()) {
                    case VertexDataType::kVertexDataTypeFloat1:
                        glVertexAttribPointer(i, 1, GL_FLOAT, false, 0, 0);
                        break;
                    case VertexDataType::kVertexDataTypeFloat2:
                        glVertexAttribPointer(i, 2, GL_FLOAT, false, 0, 0);
                        break;
                    case VertexDataType::kVertexDataTypeFloat3:
                        glVertexAttribPointer(i, 3, GL_FLOAT, false, 0, 0);
                        break;
                    case VertexDataType::kVertexDataTypeFloat4:
                        glVertexAttribPointer(i, 4, GL_FLOAT, false, 0, 0);
                        break;
                    default:
                        assert(0);
                }

                m_Buffers.push_back(buffer_id);
            }

            auto indexGroupCount = pMesh->GetIndexGroupCount();

            GLenum  mode;
            switch(pMesh->GetPrimitiveType())
            {
                case PrimitiveType::kPrimitiveTypePointList:
                    mode = GL_POINTS;
                    break;
                case PrimitiveType::kPrimitiveTypeLineList:
                    mode = GL_LINES;
                    break;
                case PrimitiveType::kPrimitiveTypeLineStrip:
                    mode = GL_LINE_STRIP;
                    break;
                case PrimitiveType::kPrimitiveTypeTriList:
                    mode = GL_TRIANGLES;
                    break;
                case PrimitiveType::kPrimitiveTypeTriStrip:
                    mode = GL_TRIANGLE_STRIP;
                    break;
                case PrimitiveType::kPrimitiveTypeTriFan:
                    mode = GL_TRIANGLE_FAN;
                    break;
                default:
                    // ignore
                    continue;
            }

            for (decltype(indexGroupCount) i = 0; i < indexGroupCount; i++)
            {
                // Generate an ID for the index buffer.
                glGenBuffers(1, &buffer_id);

                const SceneObjectIndexArray& index_array      = pMesh->GetIndexArray(i);
                auto index_array_size = index_array.GetDataSize();
                auto index_array_data = index_array.GetData();

                // Bind the index buffer and load the index data into it.
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer_id);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_array_size, index_array_data, GL_STATIC_DRAW);

                // Set the number of indices in the index array.
                GLsizei indexCount = static_cast<GLsizei>(index_array.GetIndexCount());
                GLenum type;
                switch(index_array.GetIndexType())
                {
                    case IndexDataType::kIndexDataTypeInt8:
                        type = GL_UNSIGNED_BYTE;
                        break;
                    case IndexDataType::kIndexDataTypeInt16:
                        type = GL_UNSIGNED_SHORT;
                        break;
                    case IndexDataType::kIndexDataTypeInt32:
                        type = GL_UNSIGNED_INT;
                        break;
                    default:
                        // not supported by OpenGL
                        cerr << "Error: Unsupported Index Type " << index_array << endl;
                        cerr << "Mesh: " << *pMesh << endl;
                        cerr << "Geometry: " << *pGeometry << endl;
                        continue;
                }

                m_Buffers.push_back(buffer_id);

                size_t material_index = index_array.GetMaterialIndex();
                std::string material_key = pGeometryNode->GetMaterialRef(material_index);
                auto material = scene.GetMaterial(material_key);
                if (material) {
                    auto color = material->GetBaseColor();
                    if (color.ValueMap) {
                        Image texture = color.ValueMap->GetTextureImage();
                        auto it = m_TextureIndex.find(material_key);
                        if (it == m_TextureIndex.end()) {
                            GLuint texture_id;
                            glGenTextures(1, &texture_id);
                            glActiveTexture(GL_TEXTURE0 + texture_id);
                            glBindTexture(GL_TEXTURE_2D, texture_id);
                            if(texture.bitcount == 24)
                            {
                                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture.Width, texture.Height, 
                                    0, GL_RGB, GL_UNSIGNED_BYTE, texture.data);
                            }
                            else
                            {
                                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.Width, texture.Height, 
                                    0, GL_RGBA, GL_UNSIGNED_BYTE, texture.data);
                            }
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

                            m_TextureIndex[color.ValueMap->GetName()] = texture_id;
                            m_Textures.push_back(texture_id);
                        }
                    }
                }

                DrawBatchContext& dbc = *(new DrawBatchContext);
                dbc.vao     = vao;
                dbc.mode    = mode;
                dbc.type    = type;
                dbc.count   = indexCount;
                dbc.node    = pGeometryNode;
                dbc.material = material;
                m_DrawBatchContext.push_back(std::move(dbc));
            }
        }
    }

    return;
}

void OpenGLESGraphicsManager::ClearBuffers()
{
    for (auto dbc : m_DrawBatchContext) {
        glDeleteVertexArrays(1, &dbc.vao);
    }

    m_DrawBatchContext.clear();

    for (auto buf : m_Buffers) {
        glDeleteBuffers(1, &buf);
    }

    for (auto texture : m_Textures) {
        glDeleteTextures(1, &texture);
    }

    m_Buffers.clear();
    m_Textures.clear();

}

void OpenGLESGraphicsManager::RenderBuffers()
{
    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    glUseProgram(m_shaderProgram);

    SetPerFrameShaderParameters(m_shaderProgram);

    for (auto dbc : m_DrawBatchContext)
    {
        Matrix4X4f trans;
        if (void* rigidBody = dbc.node->RigidBody()) {
            // the geometry has rigid body bounded, we blend the simlation result here.
            Matrix4X4f simulated_result = g_pPhysicsManager->GetRigidBodyTransform(rigidBody);

            BuildIdentityMatrix(trans);

            // apply the rotation part of the simlation result
            memcpy(trans[0], simulated_result[0], sizeof(float) * 3);
            memcpy(trans[1], simulated_result[1], sizeof(float) * 3);
            memcpy(trans[2], simulated_result[2], sizeof(float) * 3);

            // replace the translation part of the matrix with simlation result directly
            memcpy(trans[3], simulated_result[3], sizeof(float) * 3);

        } else {
            trans = *dbc.node->GetCalculatedTransform();
        }

        SetPerBatchShaderParameters(m_shaderProgram, "modelMatrix", trans);
        glBindVertexArray(dbc.vao);

        /* well, we have different material for each index buffer so we can not draw them together
         * in future we should group indicies according to its material and draw them together
        auto indexBufferCount = dbc.counts.size();
        const GLvoid ** pIndicies = new const GLvoid*[indexBufferCount];
        memset(pIndicies, 0x00, sizeof(GLvoid*) * indexBufferCount);
        // Render the vertex buffer using the index buffer.
        glMultiDrawElements(dbc.mode, dbc.counts.data(), dbc.type, pIndicies, indexBufferCount);
        delete[] pIndicies;
        */

        SetPerBatchShaderParameters(m_shaderProgram, "usingDiffuseMap", false);
        SetPerBatchShaderParameters(m_shaderProgram, "usingNormalMap", false);

        if (dbc.material) {
            Color color = dbc.material->GetBaseColor();
            if (color.ValueMap) {
                SetPerBatchShaderParameters(m_shaderProgram, "diffuseMap", m_TextureIndex[color.ValueMap->GetName()]);
                // set this to tell shader to use texture
                SetPerBatchShaderParameters(m_shaderProgram, "usingDiffuseMap", true);
            }
            else
            {
                SetPerBatchShaderParameters(m_shaderProgram, "diffuseColor", Vector3f({color.Value[0], color.Value[1], color.Value[2]}));
            }

            Normal normal = dbc.material->GetNormal();
            if (normal.ValueMap) {
                SetPerBatchShaderParameters(m_shaderProgram, "normalMap", m_TextureIndex[normal.ValueMap->GetName()]);
                // set this to tell shader to use texture
                SetPerBatchShaderParameters(m_shaderProgram, "usingNormalMap", true);
            }

            color = dbc.material->GetSpecularColor();
            SetPerBatchShaderParameters(m_shaderProgram, "specularColor", Vector3f({color.Value[0], color.Value[1], color.Value[2]}));

            Parameter param = dbc.material->GetSpecularPower();
            SetPerBatchShaderParameters(m_shaderProgram, "specularPower", param.Value);
        }

        glDrawElements(dbc.mode, dbc.count, dbc.type, 0x00);
    }

    return;
}

bool OpenGLESGraphicsManager::InitializeShaders()
{
    const char* vsFilename = VS_SHADER_SOURCE_FILE;
    const char* fsFilename = PS_SHADER_SOURCE_FILE;
#ifdef DEBUG
    const char* debugVsFilename = DEBUG_VS_SHADER_SOURCE_FILE;
    const char* debugFsFilename = DEBUG_PS_SHADER_SOURCE_FILE;
#endif

    std::string vertexShaderBuffer;
    std::string fragmentShaderBuffer;
    int status;

    // Load the vertex shader source file into a text buffer.
    vertexShaderBuffer = g_pAssetLoader->SyncOpenAndReadTextFileToString(vsFilename);
    if(vertexShaderBuffer.empty())
    {
            return false;
    }

    // Load the fragment shader source file into a text buffer.
    fragmentShaderBuffer = g_pAssetLoader->SyncOpenAndReadTextFileToString(fsFilename);
    if(fragmentShaderBuffer.empty())
    {
            return false;
    }

#ifdef DEBUG
    std::string debugVertexShaderBuffer;
    std::string debugFragmentShaderBuffer;

    // Load the fragment shader source file into a text buffer.
    debugVertexShaderBuffer = g_pAssetLoader->SyncOpenAndReadTextFileToString(debugVsFilename);
    if(debugVertexShaderBuffer.empty())
    {
            return false;
    }

    // Load the fragment shader source file into a text buffer.
    debugFragmentShaderBuffer = g_pAssetLoader->SyncOpenAndReadTextFileToString(debugFsFilename);
    if(debugFragmentShaderBuffer.empty())
    {
            return false;
    }
#endif

    // Create a vertex and fragment shader object.
    m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
    m_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
#ifdef DEBUG
    m_debugVertexShader = glCreateShader(GL_VERTEX_SHADER);
    m_debugFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
#endif

    // Copy the shader source code strings into the vertex and fragment shader objects.
    const char* _v_c_str =  vertexShaderBuffer.c_str();
    glShaderSource(m_vertexShader, 1, &_v_c_str, NULL);
    const char* _f_c_str =  fragmentShaderBuffer.c_str();
    glShaderSource(m_fragmentShader, 1, &_f_c_str, NULL);
#ifdef DEBUG
    const char* _v_c_str_debug = debugVertexShaderBuffer.c_str();
    glShaderSource(m_debugVertexShader, 1, &_v_c_str_debug, NULL);
    const char* _f_c_str_debug = debugFragmentShaderBuffer.c_str();
    glShaderSource(m_debugFragmentShader, 1, &_f_c_str_debug, NULL);
#endif

    // Compile the shaders.
    glCompileShader(m_vertexShader);
    glCompileShader(m_fragmentShader);
#ifdef DEBUG
    glCompileShader(m_debugVertexShader);
    glCompileShader(m_debugFragmentShader);
#endif

    // Check to see if the vertex shader compiled successfully.
    glGetShaderiv(m_vertexShader, GL_COMPILE_STATUS, &status);
    if(status != 1)
    {
            // If it did not compile then write the syntax error message out to a text file for review.
            OutputShaderErrorMessage(m_vertexShader, vsFilename);
            return false;
    }

    // Check to see if the fragment shader compiled successfully.
    glGetShaderiv(m_fragmentShader, GL_COMPILE_STATUS, &status);
    if(status != 1)
    {
            // If it did not compile then write the syntax error message out to a text file for review.
            OutputShaderErrorMessage(m_fragmentShader, fsFilename);
            return false;
    }

#ifdef DEBUG
    // Check to see if the fragment shader compiled successfully.
    glGetShaderiv(m_debugVertexShader, GL_COMPILE_STATUS, &status);
    if(status != 1)
    {
            // If it did not compile then write the syntax error message out to a text file for review.
            OutputShaderErrorMessage(m_debugVertexShader, debugVsFilename);
            return false;
    }

    // Check to see if the fragment shader compiled successfully.
    glGetShaderiv(m_debugFragmentShader, GL_COMPILE_STATUS, &status);
    if(status != 1)
    {
            // If it did not compile then write the syntax error message out to a text file for review.
            OutputShaderErrorMessage(m_debugFragmentShader, debugFsFilename);
            return false;
    }
#endif

    // Create a shader program object.
    m_shaderProgram = glCreateProgram();
#ifdef DEBUG
    m_debugShaderProgram = glCreateProgram();
#endif

    // Attach the vertex and fragment shader to the program object.
    glAttachShader(m_shaderProgram, m_vertexShader);
    glAttachShader(m_shaderProgram, m_fragmentShader);
#ifdef DEBUG
    glAttachShader(m_debugShaderProgram, m_debugVertexShader);
    glAttachShader(m_debugShaderProgram, m_debugFragmentShader);
#endif

    // Bind the shader input variables.
    glBindAttribLocation(m_shaderProgram, 0, "inputPosition");
    glBindAttribLocation(m_shaderProgram, 1, "inputNormal");
    glBindAttribLocation(m_shaderProgram, 2, "inputUV");

    // Link the shader program.
    glLinkProgram(m_shaderProgram);

#ifdef DEBUG
    // Bind the shader input variables.
    glBindAttribLocation(m_debugShaderProgram, 0, "inputPosition");

    // Link the shader program.
    glLinkProgram(m_debugShaderProgram);
#endif

    // Check the status of the link.
    glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &status);
    if(status != 1)
    {
            // If it did not link then write the syntax error message out to a text file for review.
            OutputLinkerErrorMessage(m_shaderProgram);
            return false;
    }

#ifdef DEBUG
    // Check the status of the link.
    glGetProgramiv(m_debugShaderProgram, GL_LINK_STATUS, &status);
    if(status != 1)
    {
            // If it did not link then write the syntax error message out to a text file for review.
            OutputLinkerErrorMessage(m_debugShaderProgram);
            return false;
    }
#endif

    return true;
}

void OpenGLESGraphicsManager::ClearShaders()
{
    if (m_shaderProgram) {
        if (m_vertexShader)
        {
            // Detach the vertex shaders from the program.
            glDetachShader(m_shaderProgram, m_vertexShader);
            // Delete the vertex shaders.
            glDeleteShader(m_vertexShader);
        }

        if (m_fragmentShader)
        {
            // Detach the fragment shaders from the program.
            glDetachShader(m_shaderProgram, m_fragmentShader);
            // Delete the fragment shaders.
            glDeleteShader(m_fragmentShader);
        }

        // Delete the shader program.
        glDeleteProgram(m_shaderProgram);
    }
}

#ifdef DEBUG

void OpenGLESGraphicsManager::DrawPoint(const Point &point, const Vector3f& color)
{
    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Bind the vertex array object to store all the buffers and vertex attributes we create here.
    glBindVertexArray(vao);

    GLuint buffer_id;

    // Generate an ID for the vertex buffer.
    glGenBuffers(1, &buffer_id);

    // Bind the vertex buffer and load the vertex (position and color) data into the vertex buffer.
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Point), point, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);

    m_DebugBuffers.push_back(buffer_id);

    DebugDrawBatchContext& dbc = *(new DebugDrawBatchContext);
    dbc.vao     = vao;
    dbc.mode    = GL_POINTS;
    dbc.count   = 1;
    dbc.color   = color;
    BuildIdentityMatrix(dbc.trans);

    m_DebugDrawBatchContext.push_back(std::move(dbc));
}

void OpenGLESGraphicsManager::DrawPoints(const Point* buffer, const size_t count, const Matrix4X4f& trans, const Vector3f& color)
{
    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Bind the vertex array object to store all the buffers and vertex attributes we create here.
    glBindVertexArray(vao);

    GLuint buffer_id;

    // Generate an ID for the vertex buffer.
    glGenBuffers(1, &buffer_id);

    // Bind the vertex buffer and load the vertex (position and color) data into the vertex buffer.
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Point) * count, buffer, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);

    m_DebugBuffers.push_back(buffer_id);

    DebugDrawBatchContext& dbc = *(new DebugDrawBatchContext);
    dbc.vao     = vao;
    dbc.mode    = GL_POINTS;
    dbc.count   = static_cast<GLsizei>(count);
    dbc.color   = color;
    dbc.trans   = trans;

    m_DebugDrawBatchContext.push_back(std::move(dbc));
}

void OpenGLESGraphicsManager::DrawPointSet(const PointSet& point_set, const Vector3f& color)
{
    Matrix4X4f trans;
    BuildIdentityMatrix(trans);

    DrawPointSet(point_set, trans, color);
}

void OpenGLESGraphicsManager::DrawPointSet(const PointSet& point_set, const Matrix4X4f& trans, const Vector3f& color)
{
    auto count = point_set.size();
    Point* buffer = new Point[count];
    int i = 0;
    for(auto point_ptr : point_set)
    {
        buffer[i++] = *point_ptr;
    }

    DrawPoints(buffer, count, trans, color);

    delete[] buffer;
}

void OpenGLESGraphicsManager::DrawLine(const PointList& vertices, const Matrix4X4f& trans, const Vector3f& color)
{
    auto count = vertices.size();
    GLfloat* _vertices = new GLfloat[3 * count];

    for (auto i = 0; i < count; i++)
    {
        _vertices[3 * i] = vertices[i]->data[0];
        _vertices[3 * i + 1] = vertices[i]->data[1];
        _vertices[3 * i + 2] = vertices[i]->data[2];
    }

    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Bind the vertex array object to store all the buffers and vertex attributes we create here.
    glBindVertexArray(vao);

    GLuint buffer_id;

    // Generate an ID for the vertex buffer.
    glGenBuffers(1, &buffer_id);

    // Bind the vertex buffer and load the vertex (position and color) data into the vertex buffer.
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * count, _vertices, GL_STATIC_DRAW);

    delete[] _vertices;

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);

    m_DebugBuffers.push_back(buffer_id);

    DebugDrawBatchContext& dbc = *(new DebugDrawBatchContext);
    dbc.vao     = vao;
    dbc.mode    = GL_LINES;
    dbc.count   = static_cast<GLsizei>(count);
    dbc.color   = color;
    dbc.trans   = trans;

    m_DebugDrawBatchContext.push_back(std::move(dbc));
}

void OpenGLESGraphicsManager::DrawLine(const PointList& vertices, const Vector3f& color)
{
    Matrix4X4f trans;
    BuildIdentityMatrix(trans);

    DrawLine(vertices, trans, color);
}

void OpenGLESGraphicsManager::DrawLine(const Point& from, const Point& to, const Vector3f& color)
{
    PointList point_list;
    point_list.push_back(make_shared<Point>(from));
    point_list.push_back(make_shared<Point>(to));

    DrawLine(point_list, color);
}

void OpenGLESGraphicsManager::DrawTriangle(const PointList& vertices, const Vector3f& color)
{
    Matrix4X4f trans;
    BuildIdentityMatrix(trans);

    DrawTriangle(vertices, trans, color);
}

void OpenGLESGraphicsManager::DrawTriangle(const PointList& vertices, const Matrix4X4f& trans, const Vector3f& color)
{
    auto count = vertices.size();
    assert(count >= 3);

    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Bind the vertex array object to store all the buffers and vertex attributes we create here.
    glBindVertexArray(vao);

    GLuint buffer_id;

    // Generate an ID for the vertex buffer.
    glGenBuffers(1, &buffer_id);

    // Bind the vertex buffer and load the vertex (position and color) data into the vertex buffer.
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    Vector3f* data = new Vector3f[count];
    for(auto i = 0; i < count; i++)
    {
        data[i] = *vertices[i];
    }
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vector3f) * count, data, GL_STATIC_DRAW);
    delete[] data;

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);

    m_DebugBuffers.push_back(buffer_id);

    DebugDrawBatchContext& dbc = *(new DebugDrawBatchContext);
    dbc.vao     = vao;
    dbc.mode    = GL_TRIANGLES;
    dbc.count   = static_cast<GLsizei>(vertices.size());
    dbc.color   = color;
    dbc.trans   = trans;

    m_DebugDrawBatchContext.push_back(std::move(dbc));
}

void OpenGLESGraphicsManager::DrawTriangleStrip(const PointList& vertices, const Vector3f& color)
{
    auto count = vertices.size();
    assert(count >= 3);

    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Bind the vertex array object to store all the buffers and vertex attributes we create here.
    glBindVertexArray(vao);

    GLuint buffer_id;

    // Generate an ID for the vertex buffer.
    glGenBuffers(1, &buffer_id);

    // Bind the vertex buffer and load the vertex (position and color) data into the vertex buffer.
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    Vector3f* data = new Vector3f[count];
    for(auto i = 0; i < count; i++)
    {
        data[i] = *vertices[i];
    }
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vector3f) * count, data, GL_STATIC_DRAW);
    delete[] data;

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);

    m_DebugBuffers.push_back(buffer_id);

    DebugDrawBatchContext& dbc = *(new DebugDrawBatchContext);
    dbc.vao     = vao;
    dbc.mode    = GL_TRIANGLE_STRIP;
    dbc.count   = static_cast<GLsizei>(vertices.size());
    dbc.color   = color * 0.5f;

    m_DebugDrawBatchContext.push_back(std::move(dbc));
}

void OpenGLESGraphicsManager::ClearDebugBuffers()
{
    for (auto dbc : m_DebugDrawBatchContext) {
        glDeleteVertexArrays(1, &dbc.vao);
    }

    m_DebugDrawBatchContext.clear();

    for (auto buf : m_DebugBuffers) {
        glDeleteBuffers(1, &buf);
    }

    m_DebugBuffers.clear();
}

void OpenGLESGraphicsManager::RenderDebugBuffers()
{
    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    glUseProgram(m_debugShaderProgram);

    SetPerFrameShaderParameters(m_debugShaderProgram);

    for (auto dbc : m_DebugDrawBatchContext)
    {
        SetPerBatchShaderParameters(m_debugShaderProgram, "FrontColor", dbc.color);
        SetPerBatchShaderParameters(m_debugShaderProgram, "modelMatrix", dbc.trans);

        glBindVertexArray(dbc.vao);
        glDrawArrays(dbc.mode, 0x00, dbc.count);
    }
}

#endif
