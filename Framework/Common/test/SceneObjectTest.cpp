#include <iostream>
#include "SceneObject.hpp"

using namespace My;
using namespace std;

int32_t main(int32_t argc, char** argv)
{
    int32_t result = 0;
    SceneObjectMesh         soMesh;
    SceneObjectMaterial     soMaterial;
    SceneObjectOmniLight    soOmniLight;
    SceneObjectSpotLight    soSpotLight;
    SceneObjectCamera       soCamera;

    cout << soMesh << endl;
    cout << soMaterial << endl;
    cout << soOmniLight << endl;
    cout << soSpotLight << endl;
    cout << soCamera   << endl;

    return result;
}

