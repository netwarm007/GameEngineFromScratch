#include <iostream>
#include "SceneObject.hpp"

using namespace My;
using namespace std;
using namespace xg;

int32_t main(int32_t argc, char** argv)
{
    int32_t result = 0;
    SceneObjectMesh         soMesh;
    SceneObjectMaterial     soMaterial;
    SceneObjectOmniLight    soOmniLight;
    SceneObjectSpotLight    soSpotLight;
    SceneObjectOrthogonalCamera       soOrthogonalCamera;
    SceneObjectPerspectiveCamera      soPerspectiveCamera;

    cout << soMesh << endl;
    cout << soMaterial << endl;
    cout << soOmniLight << endl;
    cout << soSpotLight << endl;
    cout << soOrthogonalCamera  << endl;
    cout << soPerspectiveCamera << endl;

    return result;
}

