#include <iostream>
#include <fstream>
#include <map>
#include "AssetLoader.hpp"

using namespace My;
using namespace std;

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

    static bool LoadShaderFromFile(const char* vsFilename, const char* fsFilename, map<int, const char*> properties, GLuint& shaderProgram)
    {
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

        // Create a vertex and fragment shader object.
        auto vertexShader = glCreateShader(GL_VERTEX_SHADER);
        auto fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

        // Copy the shader source code strings into the vertex and fragment shader objects.
        const char* _v_c_str =  vertexShaderBuffer.c_str();
        glShaderSource(vertexShader, 1, &_v_c_str, NULL);
        const char* _f_c_str =  fragmentShaderBuffer.c_str();
        glShaderSource(fragmentShader, 1, &_f_c_str, NULL);

        // Compile the shaders.
        glCompileShader(vertexShader);
        glCompileShader(fragmentShader);

        // Check to see if the vertex shader compiled successfully.
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
        if(status != 1)
        {
                // If it did not compile then write the syntax error message out to a text file for review.
                OutputShaderErrorMessage(vertexShader, vsFilename);
                return false;
        }

        // Check to see if the fragment shader compiled successfully.
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
        if(status != 1)
        {
                // If it did not compile then write the syntax error message out to a text file for review.
                OutputShaderErrorMessage(fragmentShader, fsFilename);
                return false;
        }

        // Create a shader program object.
        shaderProgram = glCreateProgram();

        // Attach the vertex and fragment shader to the program object.
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);

        // Bind the shader input variables.
        for (auto property : properties)
        {
            glBindAttribLocation(shaderProgram, property.first, property.second);
        }

        // Link the shader program.
        glLinkProgram(shaderProgram);

        // Check the status of the link.
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
        if(status != 1)
        {
                // If it did not link then write the syntax error message out to a text file for review.
                OutputLinkerErrorMessage(shaderProgram);
                return false;
        }

        return true;
    }
}

int OpenGLShaderManagerCommonBase::Initialize()
{
    return InitializeShaders() == false;
}

void OpenGLShaderManagerCommonBase::Finalize()
{
    ClearShaders();
}

void OpenGLShaderManagerCommonBase::Tick()
{

}

bool OpenGLShaderManagerCommonBase::InitializeShaders()
{
    map<int, const char*> properties;
    GLuint shaderProgram;
    bool result;

    // Forward Shader
    properties = {  
                    {0, "inputPosition"},
                    {1, "inputNormal"},
                    {2, "inputUV"}
                 };

    result = LoadShaderFromFile(VS_SHADER_SOURCE_FILE, PS_SHADER_SOURCE_FILE, properties, shaderProgram);
    if (!result)
    {
        return result;
    }

    m_DefaultShaders[DefaultShaderIndex::Forward] = shaderProgram;

    // Shadow Map Shader
    properties = {  
                    {0, "inputPosition"}
                 };

    result = LoadShaderFromFile(VS_SHADOWMAP_SOURCE_FILE, PS_SHADOWMAP_SOURCE_FILE, properties, shaderProgram);
    if (!result)
    {
        return result;
    }

    m_DefaultShaders[DefaultShaderIndex::ShadowMap] = shaderProgram;

    // Texture copy shader
    properties = {  
                    {0, "inputPosition"}
                 };

    result = LoadShaderFromFile(VS_PASSTHROUGH_SOURCE_FILE, PS_SIMPLE_TEXTURE_SOURCE_FILE, properties, shaderProgram);
    if (!result)
    {
        return result;
    }

    m_DefaultShaders[DefaultShaderIndex::Copy] = shaderProgram;

#ifdef DEBUG
    // Debug Shader
    properties = {  
                    {0, "inputPosition"}
                 };

    result = LoadShaderFromFile(DEBUG_VS_SHADER_SOURCE_FILE, DEBUG_PS_SHADER_SOURCE_FILE, properties, shaderProgram);
    if (!result)
    {
        return result;
    }

    m_DefaultShaders[DefaultShaderIndex::Debug] = shaderProgram;
#endif

    return true;
}

void OpenGLShaderManagerCommonBase::ClearShaders()
{

}
