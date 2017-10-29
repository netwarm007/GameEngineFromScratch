#include <iostream>
#include "SceneObject.hpp"
#include "SceneNode.hpp"

using namespace My;
using namespace std;
using namespace xg;

int32_t main(int32_t argc, char** argv)
{
    int32_t result = 0;
    std::shared_ptr<SceneObjectGeometry>    soGeometry(new SceneObjectGeometry());
    std::shared_ptr<SceneObjectOmniLight>    soOmniLight(new SceneObjectOmniLight());
    std::shared_ptr<SceneObjectSpotLight>    soSpotLight(new SceneObjectSpotLight());
    std::shared_ptr<SceneObjectOrthogonalCamera>      soOrthogonalCamera(new SceneObjectOrthogonalCamera());
    std::shared_ptr<SceneObjectPerspectiveCamera>     soPerspectiveCamera(new SceneObjectPerspectiveCamera());

    std::shared_ptr<SceneObjectMesh>         soMesh(new SceneObjectMesh());
    std::shared_ptr<SceneObjectMaterial>     soMaterial(new SceneObjectMaterial());

    soGeometry->AddMesh(soMesh);

    cout << *soGeometry << endl;
    cout << *soMaterial << endl;
    cout << *soOmniLight << endl;
    cout << *soSpotLight << endl;
    cout << *soOrthogonalCamera  << endl;
    cout << *soPerspectiveCamera << endl;

    SceneEmptyNode      snEmpty;
    std::unique_ptr<SceneGeometryNode>   snGeometry(new SceneGeometryNode());
    std::unique_ptr<SceneLightNode>     snLight(new SceneLightNode());
    std::unique_ptr<SceneCameraNode>     snCamera(new SceneCameraNode());

    snGeometry->AddSceneObjectRef(soGeometry);
    snLight->AddSceneObjectRef(soSpotLight);
    snCamera->AddSceneObjectRef(soOrthogonalCamera);

    snEmpty.AppendChild(std::move(snGeometry));
    snEmpty.AppendChild(std::move(snLight));
    snEmpty.AppendChild(std::move(snCamera));

    cout << snEmpty << endl;

    return result;
}

