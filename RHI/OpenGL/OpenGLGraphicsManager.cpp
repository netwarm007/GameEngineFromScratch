#include <iostream>
#include <fstream>
#include "OpenGLGraphicsManager.hpp"
#include "AssetLoader.hpp"
#include "glad/glad.h"

const char VS_SHADER_SOURCE_FILE[] = "Shaders/color.vs";
const char PS_SHADER_SOURCE_FILE[] = "Shaders/color.ps";

using namespace My;
using namespace std;

extern struct gladGLversionStruct GLVersion;

int OpenGLGraphicsManager::Initialize()
{
    int result;

    result = gladLoadGL();
    if (!result) {
        cerr << "OpenGL load failed!" << endl;
        result = -1;
    } else {
        result = 0;
        cout << "OpenGL Version " << GLVersion.major << "." << GLVersion.minor << " loaded" << endl;

        if (GLAD_GL_VERSION_3_0) {
            // Set the depth buffer to be entirely cleared to 1.0 values.
            glClearDepth(1.0f);

            // Enable depth testing.
            glEnable(GL_DEPTH_TEST);

            // Set the polygon winding to front facing for the left handed system.
            glFrontFace(GL_CW);

            // Enable back face culling.
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }

        InitializeShader(VS_SHADER_SOURCE_FILE, PS_SHADER_SOURCE_FILE);
    }

    return result;
}

void OpenGLGraphicsManager::Finalize()
{
}

void OpenGLGraphicsManager::Tick()
{
}


namespace My {
    extern AssetLoader* g_pAssetLoader;

    static void OutputShaderErrorMessage(unsigned int shaderId, const char* shaderFilename)
    {
            int logSize, i;
            char* infoLog;
            ofstream fout;
            unsigned int error;

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

bool OpenGLGraphicsManager::InitializeShader(const char* vsFilename, const char* fsFilename)
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
    m_VertexShader = glCreateShader(GL_VERTEX_SHADER);
    m_FragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    // Copy the shader source code strings into the vertex and fragment shader objects.
    const char* _v_c_str =  vertexShaderBuffer.c_str();
    glShaderSource(m_VertexShader, 1, &_v_c_str, NULL);
    const char* _f_c_str =  fragmentShaderBuffer.c_str();
    glShaderSource(m_FragmentShader, 1, &_f_c_str, NULL);

    // Compile the shaders.
    glCompileShader(m_VertexShader);
    glCompileShader(m_FragmentShader);

    // Check to see if the vertex shader compiled successfully.
    glGetShaderiv(m_VertexShader, GL_COMPILE_STATUS, &status);
    if(status != 1)
    {
            // If it did not compile then write the syntax error message out to a text file for review.
            OutputShaderErrorMessage(m_VertexShader, vsFilename);
            return false;
    }

    // Check to see if the fragment shader compiled successfully.
    glGetShaderiv(m_FragmentShader, GL_COMPILE_STATUS, &status);
    if(status != 1)
    {
            // If it did not compile then write the syntax error message out to a text file for review.
            OutputShaderErrorMessage(m_FragmentShader, fsFilename);
            return false;
    }

    // Create a shader program object.
    m_ShaderProgram = glCreateProgram();

    // Attach the vertex and fragment shader to the program object.
    glAttachShader(m_ShaderProgram, m_VertexShader);
    glAttachShader(m_ShaderProgram, m_FragmentShader);

    // Bind the shader input variables.
    glBindAttribLocation(m_ShaderProgram, 0, "inputPosition");
    glBindAttribLocation(m_ShaderProgram, 1, "inputColor");

    // Link the shader program.
    glLinkProgram(m_ShaderProgram);

    // Check the status of the link.
    glGetProgramiv(m_ShaderProgram, GL_LINK_STATUS, &status);
    if(status != 1)
    {
            // If it did not link then write the syntax error message out to a text file for review.
            OutputLinkerErrorMessage(m_ShaderProgram);
            return false;
    }

    return true;
}

