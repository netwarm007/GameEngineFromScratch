#include <iostream>
#include "GameLogic.hpp"

using namespace My;
using namespace std;

int GameLogic::Initialize()
{
    return 0;
}

void GameLogic::Finalize()
{

}

void GameLogic::Tick()
{

}

void GameLogic::OnUpKeyDown()
{
#ifdef DEBUG
    cerr << "[GameLogic] OnUpKeyDown()" << endl;
#endif
}

void GameLogic::OnUpKeyUp()
{
#ifdef DEBUG
    cerr << "[GameLogic] OnUpKeyUp()" << endl;
#endif
}

void GameLogic::OnUpKey()
{
#ifdef DEBUG
    cerr << "[GameLogic] OnUpKey()" << endl;
#endif
}

void GameLogic::OnDownKeyDown()
{
#ifdef DEBUG
    cerr << "[GameLogic] OnDownKeyDown()" << endl;
#endif
}

void GameLogic::OnDownKeyUp()
{
#ifdef DEBUG
    cerr << "[GameLogic] OnDownKeyUp()" << endl;
#endif
}

void GameLogic::OnDownKey()
{
#ifdef DEBUG
    cerr << "[GameLogic] OnDownKey()" << endl;
#endif
}

void GameLogic::OnLeftKeyDown()
{
#ifdef DEBUG
    cerr << "[GameLogic] OnLeftKeyDown()" << endl;
#endif
}

void GameLogic::OnLeftKeyUp()
{
#ifdef DEBUG
    cerr << "[GameLogic] OnLeftKeyUp()" << endl;
#endif
}

void GameLogic::OnLeftKey()
{
#ifdef DEBUG
    cerr << "[GameLogic] OnLeftKey()" << endl;
#endif
}

void GameLogic::OnRightKeyDown()
{
#ifdef DEBUG
    cerr << "[GameLogic] OnRightKeyDown()" << endl;
#endif
}

void GameLogic::OnRightKeyUp()
{
#ifdef DEBUG
    cerr << "[GameLogic] OnRightKeyUp()" << endl;
#endif
}

void GameLogic::OnRightKey()
{
#ifdef DEBUG
    cerr << "[GameLogic] OnRightKey()" << endl;
#endif
}
