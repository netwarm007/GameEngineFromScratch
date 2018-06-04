#include <iostream>
#include <fstream>
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

void OpenGLShaderManagerCommonBase::ClearShaders()
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

void* OpenGLShaderManagerCommonBase::GetDefaultShaderProgram()
{
    return static_cast<void*>(&m_shaderProgram);
}

#ifdef DEBUG
void* OpenGLShaderManagerCommonBase::GetDebugShaderProgram()
{
    return static_cast<void*>(&m_debugShaderProgram);
}
#endif
