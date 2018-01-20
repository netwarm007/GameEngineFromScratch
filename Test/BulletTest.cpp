#include <iostream>
#define BT_USE_DOUBLE_PRECISION 1
#include <btBulletDynamicsCommon.h>

using namespace std;

int main(int, char**)
{
    // Build the broadphase
    btBroadphaseInterface* broadphase = new btDbvtBroadphase();

    // Set up the collision configuration and dispatcher
    btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();
    btCollisionDispatcher* dispatcher = new btCollisionDispatcher(collisionConfiguration);

    // The actual physics solver
    btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;

    // The world
    btDiscreteDynamicsWorld* dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration);
    dynamicsWorld->setGravity(btVector3(0.0f, -9.8f, 0.0f));

    // Create Collision Models
    btCollisionShape* groundShape = new btStaticPlaneShape(btVector3(0.0f, 1.0f, 0.0f), 1);
    btCollisionShape* fallShape = new btSphereShape(1.0f);

    // Create Rigid Body
    btDefaultMotionState* groundMotionState =        new btDefaultMotionState(btTransform(btQuaternion(0.0f, 0.0f, 0.0f, 1.0f), btVector3(0.0f, -1.0f, 0.0f)));
    btRigidBody::btRigidBodyConstructionInfo
        groundRigidBodyCI(0.0f, groundMotionState, groundShape, btVector3(0.0f, 0.0f, 0.0f));
    btRigidBody* groundRigidBody = new btRigidBody(groundRigidBodyCI);
    dynamicsWorld->addRigidBody(groundRigidBody);

    btDefaultMotionState* fallMotionState = 
        new btDefaultMotionState(btTransform(btQuaternion(0.0f, 0.0f, 0.0f, 1.0f), btVector3(0.0f, 50.0f, 0.0f)));
    btScalar mass = 1.0f;
    btVector3 fallInertia(0.0f, 0.0f, 0.0f);
    fallShape->calculateLocalInertia(mass, fallInertia);
    btRigidBody::btRigidBodyConstructionInfo
        fallRigidBodyCI(mass, fallMotionState, fallShape, fallInertia);
    btRigidBody* fallRigidBody = new btRigidBody(fallRigidBodyCI);
    dynamicsWorld->addRigidBody(fallRigidBody);

    for (int i = 0; i < 300; i++) {
        dynamicsWorld->stepSimulation(1.0f / 60.0f, 10.0f);

        btTransform trans;
        fallRigidBody->getMotionState()->getWorldTransform(trans);

        cout << "Sphere Height: " << trans.getOrigin().getY() << endl;
    }

    dynamicsWorld->removeRigidBody(fallRigidBody);
    delete fallRigidBody->getMotionState();
    delete fallRigidBody;

    dynamicsWorld->removeRigidBody(groundRigidBody);
    delete groundRigidBody->getMotionState();
    delete groundRigidBody;

    delete fallShape;
    delete groundShape;

    // Clean up
    delete dynamicsWorld;
    delete solver;
    delete dispatcher;
    delete collisionConfiguration;
    delete broadphase;

    return 0;
}

