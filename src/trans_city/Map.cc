/**
 * \file Map.cc
 * \brief The map for the game engine
 */

#include <random>
#include <queue>
#include "TransCity.h"
#include "Map.h"

namespace magent {
namespace trans_city {

Map::Map() : slots(nullptr) {
}

Map::~Map() {
    delete [] slots;
}

void Map::reset(std::vector<Position> &walls, int width, int height) {
    if (slots != nullptr)
        delete [] slots;
    slots = new Slot[width * height];

    for (int i = 0; i < width * height; i++) {
        slots[i].occ_type = OCC_NONE;
        slots[i].mask = 0;
    }

    map_width = width;
    map_height = height;

    // init border
    for (int i = 0; i < map_width; i++) {
        add_wall(Position{i, 0});
        add_wall(Position{i, map_height-1});
        walls.push_back(Position{i, 0});
        walls.push_back(Position{i, map_height-1});
    }
    for (int i = 0; i < map_height; i++) {
        add_wall(Position{0, i});
        add_wall(Position{map_width-1, i});
        walls.push_back(Position{0, i});
        walls.push_back(Position{map_height-1, i});
    }
}

void Map::init_mask(std::vector<Road> &roads, std::vector<TrafficLight> &lights) {
    for (auto road : roads) {
        int dir, x0, y0, w, h;
        std::tie(x0, y0, w, h) = road.get_location();
        dir = road.get_dir();

        if (dir == 0) {
            for (int x = x0; x < x0 + w; x++) {
                slots[pos2int(x, y0)].mask = ~((1 << 5) | (1 << 6) | (1 << 0));
                for (int y = y0 + 1; y < y0 + h/2; y++) {
                    slots[pos2int(x, y)].mask = ~((1 << 5) | (1 << 6) | (1 << 7));
                }
                for (int y = y0 + h/2; y < y0 + h-1; y++) {
                    slots[pos2int(x, y)].mask = ~((1 << 1) | (1 << 2) | (1 << 3));
                }
                slots[pos2int(x, y0 + h-1)].mask = ~((1 << 1) | (1 << 2) | (1 << 4));
            }
        } else {
            for (int y = y0; y < y0 + h; y++) {
                slots[pos2int(x0, y)].mask = ~((1 << 3) | (1 << 4) | (1 << 6));
                for (int x = x0 + 1; x < x0 + w/2; x++) {
                    slots[pos2int(x, y)].mask = ~((1 << 3) | (1 << 4) | (1 << 5));
                }
                for (int x = x0 + w/2; x < x0 + w - 1; x++) {
                    slots[pos2int(x, y)].mask = ~((1 << 0) | (1 << 7) | (1 << 1));
                }
                slots[pos2int(x0 + w-1, y)].mask = ~((1 << 0) | (1 << 7) | (1 << 2));
            }
        }
    }

    for (auto light : lights) {
        int x0, y0, w, h;
        std::tie(x0, y0, w, h) = light.get_location();

        for (int x = x0 + 1; x < x0 + w; x++) {
            if (x - x0 <= w / 2) {
                slots[pos2int(x, y0)].mask = ~(1 << 4);
                slots[pos2int(x, y0 +h)].mask = ~(1 << 4);
            } else {
                slots[pos2int(x, y0)].mask = ~(1 << 0);
                slots[pos2int(x, y0 +h)].mask = ~(1 << 0);
            }
        }

        for (int y = y0 + 1; y < y0 + h; y++) {
            if (y - y0 <= h / 2) {
                slots[pos2int(x0, y)].mask = ~(1 << 6);
                slots[pos2int(x0 + w, y)].mask = ~(1 << 6);
            } else {
                slots[pos2int(x0, y)].mask = ~(1 << 2);
                slots[pos2int(x0 + w, y)].mask = ~(1 << 2);
            }
        }
    }
}

Position Map::get_random_blank(std::default_random_engine &random_engine) {
    int tries = 0;
    while (true) {
        int x = (int) random_engine() % (map_width);
        int y = (int) random_engine() % (map_height);

        if (slots[pos2int(x, y)].occ_type == OCC_NONE) {
            return Position{x, y};
        }

        if (tries++ > map_width * map_height) {
            LOG(FATAL) << "cannot find a blank position in a filled map";
        }
    }
}

int Map::add_agent(Agent *agent) {
    PositionInteger pos_int = pos2int(agent->get_pos());
    if (slots[pos_int].occ_type != OCC_NONE)
        return 0;
    slots[pos_int].occ_type = OCC_AGENT;
    slots[pos_int].occupier = agent;
    slots[pos_int].occ_ct = 1;
    return 1;
}

int Map::add_wall(Position pos) {
    PositionInteger pos_int = pos2int(pos);
    if (slots[pos_int].occ_type != OCC_NONE)
        return 0;
    slots[pos_int].occ_type = OCC_WALL;
    return 1;
}

int Map::add_light(Position pos, int w, int h) {
    int x = pos.x;
    int y = pos.y;

    Position poss[] = {
            Position{x, y},
            Position{x, y+h},
            Position{x+w, y},
            Position{x+w, y+h}
    };

    slots[pos2int(x, y)].occ_type = OCC_LIGHT;
    slots[pos2int(x, y+h)].occ_type = OCC_LIGHT;
    slots[pos2int(x+w, y)].occ_type = OCC_LIGHT;
    slots[pos2int(x+w, y+h)].occ_type = OCC_LIGHT;

    return 1;
}

int Map::add_park(Position pos, int w, int h, int no) {
    for (int x = pos.x; x < pos.x + w; x++)
        for (int y = pos.y; y < pos.y + h; y++) {
            slots[pos2int(x, y)].occ_type = OCC_PARK;
            slots[pos2int(x, y)].occupier = reinterpret_cast<void*>(no);
        }
    return 1;
}

void Map::extract_view(const Agent* agent, float *linear_buffer, int height, int width, int channel) {
    Position pos = agent->get_pos();

    NDPointer<float, 3> buffer(linear_buffer, {{height, width, channel}});

    int x_start = pos.x - width / 2;
    int y_start = pos.y - height / 2;
    int x_end = x_start + width - 1;
    int y_end = y_start + height - 1;

    x_start = std::max(0, std::min(map_width-1, x_start));
    x_end   = std::max(0, std::min(map_width-1, x_end));
    y_start = std::max(0, std::min(map_height-1, y_start));
    y_end   = std::max(0, std::min(map_height-1, y_end));

    int view_x_start = 0 + x_start - (pos.x - width/2);
    int view_y_start = 0 + y_start - (pos.y - height/2);
    int view_x = view_x_start;
    for (int x = x_start; x <= x_end; x++) {
        int view_y = view_y_start;
        for (int y =  y_start; y <= y_end; y++) {
            PositionInteger pos_int = pos2int(x, y);
            Agent *occupier;
            int color;
            switch (slots[pos_int].occ_type) {
                case OCC_NONE:
                    break;
                case OCC_WALL:
                    buffer.at(view_y, view_x, CHANNEL_WALL) = 1;
                    break;
                case OCC_LIGHT:
                    buffer.at(view_y, view_x, CHANNEL_LIGHT) = 1;
                    break;
                case OCC_PARK:
                    if (reinterpret_cast<long>(slots[pos_int].occupier) == agent->get_color()) {
                        buffer.at(view_y, view_x, CHANNEL_PARK_SELF) = 1;
                    } else {
                        buffer.at(view_y, view_x, CHANNEL_PARK_OTHER) = 1;
                    }
                    break;
                case OCC_AGENT:
                    if ((Agent*)slots[pos_int].occupier == agent) {
                        buffer.at(view_y, view_x, CHANNEL_SELF) = 1;
                    } else {
                        buffer.at(view_y, view_x, CHANNEL_OTHER) = 1;
                    }
                    break;
            }
            view_y++;
        }
        view_x++;
    }
}

int Map::do_move(Agent *agent, int dir,
                 const std::map<std::pair<Position, Position>, TrafficLine> &lines,
                 const std::vector<TrafficLight> &lights) {
    static const int delta[][2] = {
            {0, -1}, {1, -1}, {1, 0}, {1, 1},
            {0, 1}, {-1, 1}, {-1, 0}, {-1, -1},
    };

    Position pos = agent->get_pos();
    Position new_pos = pos;

    if (slots[pos2int(pos)].mask & (1 << dir)) {
        return -4;
    }

    new_pos.x += delta[dir][0];
    new_pos.y += delta[dir][1];

    if (slots[pos2int(new_pos)].occ_type == OCC_NONE) {
        auto iter = lines.find(std::make_pair(pos, new_pos));
        if (iter != lines.end() && !iter->second.can_pass(lights))
            return -1;

        slots[pos2int(new_pos)].occ_type = OCC_AGENT;
        slots[pos2int(new_pos)].occupier = agent;
        slots[pos2int(pos)].occ_type = OCC_NONE;
        agent->set_pos(new_pos);
        return 0;
    } else if (slots[pos2int(new_pos)].occ_type == OCC_PARK) {
        if (reinterpret_cast<long>(slots[pos2int(new_pos)].occupier) == agent->get_color()) {
            slots[pos2int(pos)].occ_type = OCC_NONE;
            return 1;
        }
        else {
            return -2;
        }
    } else {
        return -3;
    }
}

struct QueueItem {
    Position pos;
    int dis;
};

void Map::calc_bfs(Park &source, std::vector<int> &ret) {
    assert(ret.size() == map_height * map_height);

    std::queue<QueueItem> que;
    std::vector<bool> visited(map_height * map_width, false);

    int x0, y0, w, h;
    std::tie(x0, y0, w, h) = source.get_location();

    for (int x = x0 - 1; x < x0 + w + 1; x++) {
        if (slots[pos2int(x, y0-1)].occ_type == OCC_NONE) {
            que.push(QueueItem{Position{x, y0-1}, 1});
            visited[pos2int(x, y0-1)] = true;
        }
        if (slots[pos2int(x, y0+h)].occ_type == OCC_NONE) {
            que.push(QueueItem{Position{x, y0+h}, 1});
            visited[pos2int(x, y0+h)] = true;
        }
    }

    for (int y = y0 - 1; y < y0 + h + 1; y++) {
        if (slots[pos2int(x0-1, y)].occ_type == OCC_NONE) {
            que.push(QueueItem{Position{x0-1, y}, 1});
            visited[pos2int(x0-1, y)] = true;
        }
        if (slots[pos2int(x0+w, y)].occ_type == OCC_NONE) {
            que.push(QueueItem{Position{x0+w, y}, 1});
            visited[pos2int(x0+w, y)] = true;
        }
    }

    const int dir[][2] = {
            {0, 1}, {1, 0}, {0, -1}, {-1, 0},
            {1, 1}, {-1, -1}, {1, -1}, {-1, 1}
    };

    while (!que.empty()) {
        QueueItem item = que.front();
        que.pop();
        int x, y, dis;
        int new_x, new_y;
        x = item.pos.x; y = item.pos.y; dis = item.dis;

        ret[pos2int(item.pos)] = dis;
        for (int i = 0; i < 8; i++) {
            new_x = x + dir[i][0];
            new_y = y + dir[i][1];
            Position new_pos{new_x, new_y};

            if (in_board(new_x, new_y) && !visited[pos2int(new_pos)]
                && slots[pos2int(new_pos)].occ_type == OCC_NONE) {
                que.push(QueueItem{new_pos, dis+1});
                visited[pos2int(new_pos)] = true;
            }
        }
    }
}


} // namespace trans_city
} // namespace magent
