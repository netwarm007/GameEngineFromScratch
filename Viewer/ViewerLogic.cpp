#include "ViewerLogic.hpp"

#include "BaseApplication.hpp"
#include "SceneManager.hpp"
#include "geommath.hpp"

#include <string>

using namespace My;
using namespace std;

int ViewerLogic::Initialize() {
    int result;

    auto pSceneManager =
        dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();

    cout << "[ViewerLogic] Viewer Logic Initialize" << endl;

    // options such as "-screenshot <path>" are consumed elsewhere; skip them
    std::string scene_filename;
    for (int i = 1; i < m_pApp->GetCommandLineArgumentsCount(); i++) {
        std::string arg = m_pApp->GetCommandLineArgument(i);
        if (arg == "-sceneCycle") {
            if (++i >= m_pApp->GetCommandLineArgumentsCount()) {
                cerr << "[ViewerLogic] -sceneCycle requires 0, 1 or 2\n";
                return -1;
            }
            const std::string phase = m_pApp->GetCommandLineArgument(i);
            if (phase != "0" && phase != "1" && phase != "2") {
                cerr << "[ViewerLogic] Invalid -sceneCycle phase\n";
                return -1;
            }
            m_SceneCycle = phase[0] - '0';
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            i++;
            continue;
        }
        scene_filename = arg;
        break;
    }

    if (!scene_filename.empty()) {
        cout << "[ViewerLogic] Loading Scene: " << scene_filename << endl;
        result = pSceneManager->LoadScene(scene_filename.c_str());
    } else {
        cout << "[ViewerLogic] Loading Splash Scene" << endl;
        result = pSceneManager->LoadScene("Scene/splash.ogex");
    }

    return result;
}

void ViewerLogic::Finalize() { cout << "[ViewerLogic] Finalize" << endl; }

void ViewerLogic::Tick() {
    if (m_SceneCycle == 0) return;
    ++m_SceneCycleTick;
    const char* scene = nullptr;
    if (m_SceneCycleTick == 8) scene = "Scene/billiard.ogex";
    if (m_SceneCycleTick == 16 && m_SceneCycle == 2)
        scene = "Scene/splash.ogex";
    if (scene) {
        auto* scenes = dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();
        if (scenes->LoadScene(scene) != 0) {
            cerr << "[ViewerLogic] Scene-cycle load failed: " << scene << '\n';
            m_pApp->RequestQuit();
            return;
        }
        cout << "SCENE_CYCLE_LOADED " << scene << endl;
    }
    if (m_SceneCycleTick == (m_SceneCycle == 1 ? 16 : 24)) {
        cout << "SCENE_CYCLE_READY phase=" << m_SceneCycle << endl;
    }
}

void ViewerLogic::OnLeftKeyDown() {
    auto pSceneManager =
        dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();

    auto& scene = pSceneManager->GetSceneForRendering();
    if (scene) {
        auto pCameraNode = scene->GetFirstCameraNode();
        if (pCameraNode) {
            auto local_axis = pCameraNode->GetLocalAxis();
            Vector3f camera_x_axis = local_axis[0];

            // pan camera along its local axis -x direction
            pCameraNode->MoveBy(camera_x_axis * -1.0f);
            auto target = pCameraNode->GetTarget();
            target -= camera_x_axis;
            pCameraNode->SetTarget(target);
        }
    }
}

void ViewerLogic::OnRightKeyDown() {
    auto pSceneManager =
        dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();

    auto& scene = pSceneManager->GetSceneForRendering();
    if (scene) {
        auto pCameraNode = scene->GetFirstCameraNode();
        if (pCameraNode) {
            auto local_axis = pCameraNode->GetLocalAxis();
            Vector3f camera_x_axis = local_axis[0];

            // pan along camera local axis x direction
            pCameraNode->MoveBy(camera_x_axis);
            auto target = pCameraNode->GetTarget();
            target += camera_x_axis;
            pCameraNode->SetTarget(target);
        }
    }
}

void ViewerLogic::OnUpKeyDown() {
    auto pSceneManager =
        dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();

    auto& scene = pSceneManager->GetSceneForRendering();
    if (scene) {
        auto pCameraNode = scene->GetFirstCameraNode();
        if (pCameraNode) {
            auto local_axis = pCameraNode->GetLocalAxis();
            Vector3f camera_y_axis = local_axis[1];

            // lifting camera along its local axis y direction
            pCameraNode->MoveBy(camera_y_axis);
            auto target = pCameraNode->GetTarget();
            target += camera_y_axis;
            pCameraNode->SetTarget(target);
        }
    }
}

void ViewerLogic::OnDownKeyDown() {
    auto pSceneManager =
        dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();

    auto& scene = pSceneManager->GetSceneForRendering();
    if (scene) {
        auto pCameraNode = scene->GetFirstCameraNode();
        if (pCameraNode) {
            auto local_axis = pCameraNode->GetLocalAxis();
            Vector3f camera_y_axis = local_axis[1];

            // lifting camera along its local axis -y direction
            pCameraNode->MoveBy(camera_y_axis * -1.0f);
            auto target = pCameraNode->GetTarget();
            target -= camera_y_axis;
            pCameraNode->SetTarget(target);
        }
    }
}

void ViewerLogic::OnAnalogStick(int id, float deltaX, float deltaY) {
    auto pSceneManager =
        dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();

    if (id == 1) {
        auto& scene = pSceneManager->GetSceneForRendering();
        if (scene) {
            auto pCameraNode = scene->GetFirstCameraNode();
            if (pCameraNode) {
                pCameraNode->RotateBy(deltaY / (-1800 * 2.0 * PI), 0.0f, 0.0f);
                pCameraNode->RotateBy(0.0f, 0.0f, deltaX / (-1800 * 2.0 * PI));
            }
        }
    }
}
