/**
 * \file city_def.h
 * \brief some global definition for discrete_snake
 */

#ifndef MAGENT_TRANSCITY_CITY_DEF_H
#define MAGENT_TRANSCITY_CITY_DEF_H

#include "../utility/utility.h"
#include "../Environment.h"

namespace magent {
namespace trans_city {

struct Position {
    int x;
    int y;

    bool operator < (const Position &other) const {
        return x < other.x || ((x == other.x) && y < other.y);
    }

    bool operator == (const Position &other) const {
        return x == other.x && y == other.y;
    }
};

typedef int PositionInteger;

typedef enum {ACT_UP, ACT_RIGHTUP, ACT_RIGHT, ACT_RIGHTDOWN, ACT_DOWN,
              ACT_LEFTDOWN, ACT_LEFT, ACT_LEFTUP, ACT_NOP, ACT_NUM} Action;

enum {CHANNEL_WALL, CHANNEL_LIGHT, CHANNEL_PARK_SELF, CHANNEL_PARK_OTHER, CHANNEL_SELF, CHANNEL_OTHER, CHANNEL_NUM};

const int MAX_COLOR_NUM = 20;

typedef float Reward;

class Agent;
class TrafficLight;
class TrafficLine;
class Park;
class Road;
class Building;

using ::magent::environment::Environment;
using ::magent::environment::GroupHandle;
using ::magent::utility::strequ;
using ::magent::utility::NDPointer;

} // namespace trans_city
} // namespace magent

#endif //MAGENT_TRANSCITY_CITY_DEF_H
