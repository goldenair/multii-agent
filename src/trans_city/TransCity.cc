/**
 * \file TransCity.cc
 * \brief core game engine of the trans_city
 */

#include <stdexcept>
#include <cstring>
#include <iostream>

#include "TransCity.h"

namespace magent {
namespace trans_city {

TransCity::TransCity() {
    width = height = 100;
    embedding_size = 16;

    random_engine.seed(0);
    interval_min = 10;
    interval_max = 20;

    reward_scale = 2;
    ban_penalty = -0.3f;
}

TransCity::~TransCity() {
    for (auto agent : agents)
        delete agent;
}

void TransCity::reset() {
    id_counter = 0;
    dis_inited = false;


    lights.clear();
    lines.clear();
    parks.clear();
    walls.clear();
    buildings.clear();
    distants.clear();

    map.reset(walls, width, height);
    render_generator.next_file();

    for (int i = 0; i < agents.size(); i++) {
        delete agents[i];
    }
    agents.clear();
}

void TransCity::set_config(const char *key, void *p_value) {
    float fvalue = *(float *)p_value;
    int ivalue   = *(int *)p_value;
    bool bvalue  = *(bool *)p_value;
    const char *strvalue = (const char *)p_value;

    if (strequ(key, "map_width"))
        width = ivalue;
    else if (strequ(key, "map_height"))
        height = ivalue;
    else if (strequ(key, "view_width"))
        view_width = ivalue;
    else if (strequ(key, "view_height"))
        view_height = ivalue;
    else if (strequ(key, "interval_min"))
        interval_min = ivalue;
    else if (strequ(key, "interval_max"))
        interval_max = ivalue;
    else if (strequ(key, "reward_scale"))
        reward_scale = fvalue;
    else if (strequ(key, "ban_penalty"))
        reward_scale = fvalue;

    else if (strequ(key, "embedding_size"))
        embedding_size = ivalue;
    else if (strequ(key, "render_dir"))
        render_generator.set_render("save_dir", strvalue);

    else if (strequ(key, "seed"))
        random_engine.seed(ivalue);

    else
        LOG(FATAL) << "invalid argument in TransCity::set_config: " << key;
}

void TransCity::add_object(int obj_id, int n, const char *method, const int *linear_buffer) {
    if (dis_inited)
        assert(obj_id >= 0);
    if (obj_id == -1) { // wall
        if (strequ(method, "custom")) {
            NDPointer<const int, 2> buf(linear_buffer, {{n, 2}});
            for (int i = 0; i < n; i++) {
                map.add_wall(Position{buf.at(i, 0), buf.at(i, 1)});
                walls.push_back(Position{buf.at(i, 0, buf.at(i, 1))});
            }
        }
    } else if (obj_id == -2) { // light
        if (strequ(method, "custom")) {
            NDPointer<const int, 2> buf(linear_buffer, {{n, 5}});
            for (int i = 0; i < n; i++) {
                int x = buf.at(i, 0);
                int y = buf.at(i, 1);
                int w = buf.at(i, 2);
                int h = buf.at(i, 3);
                int mask = buf.at(i, 4);
                assert(mask != 0);
                Position pos{x, y};
                map.add_light(pos, w, h);
                int interval = static_cast<int>(random_engine()) % (interval_max - interval_min) + interval_min;

                lights.emplace_back(TrafficLight(pos, buf.at(i, 2), buf.at(i,3), interval, mask));
                int idx = static_cast<int>(lights.size()) - 1;

                // up down (horizontal)
                for (int j = 1; j < w; j++) {
                    Position pos_1, pos_2;
                    if (mask & 0x1) {
                        pos_1.x = x + j; pos_1.y = y; pos_2.x = x + j; pos_2.y = y + 1;
                        lines[std::make_pair(pos_1, pos_2)] = TrafficLine(idx, 0);
                        lines[std::make_pair(pos_2, pos_1)] = TrafficLine(idx, 0);
                    }
                    if (mask & 0x4) {
                        pos_1.x = x + j; pos_1.y = y + (h - 1); pos_2.x = x + j; pos_2.y = y + (h - 1) + 1;
                        lines[std::make_pair(pos_1, pos_2)] = TrafficLine(idx, 0);
                        lines[std::make_pair(pos_2, pos_1)] = TrafficLine(idx, 0);
                    }
                }

                // left right (vertical)
                for (int j = 1; j < h; j++) {
                    Position pos_1, pos_2;
                    if (mask & 0x2) {
                        pos_1.x = x + (w - 1); pos_1.y = y + j; pos_2.x = x + (w - 1) + 1; pos_2.y = y + j;
                        lines[std::make_pair(pos_1, pos_2)] = TrafficLine(idx, 1);
                        lines[std::make_pair(pos_2, pos_1)] = TrafficLine(idx, 1);
                    }
                    if (mask & 0x8) {
                        pos_1.x = x; pos_1.y = y + j; pos_2.x = x + 1; pos_2.y = y + j;
                        lines[std::make_pair(pos_1, pos_2)] = TrafficLine(idx, 1);
                        lines[std::make_pair(pos_2, pos_1)] = TrafficLine(idx, 1);
                    }
                }
            }
        }
    } else if (obj_id == -3) { // park
        if (strequ(method, "custom")) {
            NDPointer<const int, 2> buf(linear_buffer, {{n, 4}});
            for (int i = 0; i < n; i++) {
                map.add_park(Position{buf.at(i, 0), buf.at(i, 1)}, buf.at(i, 2), buf.at(i, 3), (int)parks.size());
                parks.emplace_back(Park(Position{buf.at(i, 0), buf.at(i, 1)}, buf.at(i, 2), buf.at(i, 3)));
            }

            if (parks.size() > MAX_COLOR_NUM)
                LOG(FATAL) << "Too many parks: " << parks.size() << "/" << MAX_COLOR_NUM;
        }
    } else if (obj_id == -4) { // building
        if (strequ(method, "custom")) {
            NDPointer<const int, 2> buf(linear_buffer, {{n, 4}});
            for (int i = 0; i < n; i++) {
                int x0 = buf.at(i, 0);
                int y0 = buf.at(i, 1);
                int w = buf.at(i, 2);
                int h = buf.at(i, 3);

                for (int x = x0; x < x0 + w; x++) {
                    for (int y = y0; y < y0 + h; y++) {
                        map.add_wall(Position{x, y});
                    }
                }
                buildings.emplace_back(Building(Position{x0, y0}, w, h));
            }
        }
    } else if (obj_id == -5) { // roads
        if(strequ(method, "custom")) {
            NDPointer<const int, 2> buf(linear_buffer, {{n, 2}});
            for (int i = 0; i < n; i++) {
                int x0 = buf.at(i, 0);
                int y0 = buf.at(i, 1);
                int w = buf.at(i, 2);
                int h = buf.at(i, 3);
                int dir = buf.at(i, 4);

                roads.emplace_back(Road(Position{x0, y0}, w, h, dir));
            }
        }
    } else if (obj_id == 0) { // car
        // init distance for all parks
        num_park = static_cast<int>(parks.size());

        if (num_park <= 0 || num_park > MAX_COLOR_NUM)
            LOG(FATAL) << "invalid number of park: " << num_park;

        if (!dis_inited) {
            dis_inited = true;
            for (int i = 0; i < num_park; i++) {
                distants.emplace_back(std::vector<int>(width * height, 0));
            }
            for (int i = 0; i < num_park; i++) {
                map.calc_bfs(parks[i], distants[i]);
            }
            map.init_mask(roads, lights);
        }
        if (strequ(method, "random")) {
            Position pos;
            for (int i = 0; i < n; i++) {
                int color = static_cast<int>(random_engine()) % num_park;
                int x, y, w, h;
                std::tie(x, y, w, h) = parks[color].get_location();
                Position goal{x + w/2, y + h/2};

                Agent *agent = new Agent(id_counter, color, goal);

                pos = map.get_random_blank(random_engine);
                agent->set_pos(pos);

                map.add_agent(agent);
                agents.push_back(agent);
            }
        } else if (strequ(method, "custom")) {
            NDPointer<const int, 2> buf(linear_buffer, {{n, 2}});

            for (int i = 0; i < n; i++) {
                int color = static_cast<int>(random_engine()) % num_park;
                int x, y, w, h;
                std::tie(x, y, w, h) = parks[color].get_location();
                Position pos = Position{buf.at(i, 0), buf.at(i, 1)};
                Position goal{x + w/2, y + h/2};

                Agent *agent = new Agent(id_counter, color, goal);

                agent->set_pos(pos);

                if (map.is_blank(pos)) {
                    map.add_agent(agent);
                    agents.push_back(agent);
                }
            }
        } else {
            LOG(FATAL) << "unsupported method in TransCity::add_object: " << method;
        }
    }
}

void TransCity::get_observation(GroupHandle group, float **linear_buffers) {
    const int delta[][2] = {
            {0, -1}, {1, -1}, {1, 0}, {1, 1},
            {0, 1}, {-1, 1}, {-1, 0}, {-1, -1},
    };

    const int n_channel = CHANNEL_NUM;
    const int n_action  = static_cast<int>(ACT_NUM);
    const int feature_size = get_feature_size_(group);

    size_t agent_size = agents.size();

    // transform buffers
    NDPointer<float, 4> view_buffer(linear_buffers[0], {{-1, view_height, view_width, n_channel}});
    NDPointer<float, 2> feature_buffer(linear_buffers[1], {{-1, feature_size}});

    memset(view_buffer.data, 0, sizeof(float) * agent_size * view_height * view_width * n_channel);
    memset(feature_buffer.data, 0, sizeof(float) * agent_size * feature_size);

    #pragma omp parallel for
    for (int i = 0; i < agent_size; i++) {
        Agent *agent = agents[i];

        map.extract_view(agent, view_buffer.data + i*view_height*view_width*n_channel,
                         view_height, view_width, n_channel);

        // get non-spatial feature
        agent->get_embedding(feature_buffer.data + i*feature_size, embedding_size);
        // last action
        feature_buffer.at(i, embedding_size + agent->get_action()) = 1;
        // last reward
        feature_buffer.at(i, embedding_size + n_action) = agent->get_last_reward();

        // diff with goal
        Position pos = agent->get_pos();
        Position goal = agent->get_goal();
        int color = agent->get_color();
        feature_buffer.at(i, embedding_size + n_action + 1) = 1.0f * pos.x / width;
        feature_buffer.at(i, embedding_size + n_action + 2) = 1.0f * pos.y / height;
        feature_buffer.at(i, embedding_size + n_action + 3) = 1.0f * goal.x / width;
        feature_buffer.at(i, embedding_size + n_action + 4) = 1.0f * goal.y / height;

        float scale = 1.0f;
        feature_buffer.at(i, embedding_size + n_action + 5) =
                scale * (distants[color][map.pos2int(pos)]) / (width + height);
        int base_dis = distants[color][map.pos2int(pos)];
        for (int j = 0; j < 8; j++) {
            Position new_pos{pos.x + delta[j][0], pos.y + delta[j][1]};
            assert (map.in_board(pos.x, pos.y));
            feature_buffer.at(i, embedding_size + n_action + 6 + j) =
                    scale * (distants[color][map.pos2int(new_pos)] - base_dis);
        }
    }
}

void TransCity::set_action(GroupHandle group, const int *actions) {
    size_t size = agents.size();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        Agent *agent = agents[i];
        Action act = (Action) actions[i];
        agent->set_action(act);
    }
}

void TransCity::step(int *done) {
    LOG(TRACE) << "step begin. ";

    LOG(TRACE) << "update lights. ";
    for (int i = 0; i < lights.size(); i++) {
        lights[i].update_status();
    }

    LOG(TRACE) << "go";
    size_t agent_size = agents.size();
    for (int i = 0; i < agent_size; i++) {
        Agent *agent = agents[i];
        Position pos = agent->get_pos();
        Position goal = agent->get_goal();

        Action act = agent->get_action();

        agent->add_reward(static_cast<float>(reward_scale *
                (-fabs(pos.x - goal.x) - fabs(pos.y - goal.y)) / (width + height)) - 0.5f * reward_scale);

        if (act >= ACT_NOP)
            continue;
        int dir = static_cast<int>(act);

        int ret = map.do_move(agent, dir, lines, lights);
        if (ret == 1) { // park
            agent->add_reward(2 * reward_scale);
            agent->set_dead();
        }
        if (ret == -4) { // banned road
            agent->add_reward(ban_penalty);
        }
    }

    *done = 0;
}

void TransCity::get_reward(GroupHandle group, float *buffer) {
    size_t agent_size = agents.size();
    #pragma omp parallel for
    for (int i = 0; i < agent_size; i++) {
        buffer[i] = agents[i]->get_reward();
    }
}

void TransCity::clear_dead() {
    size_t pt = 0;
    size_t agent_size = agents.size();
    for (int j = 0; j < agent_size; j++) {
        Agent *agent = agents[j];
        if (agent->is_dead()) {
            delete agent;
        } else {
            agent->init_reward();
            agents[pt++] = agent;
        }
    }
    agents.resize(pt);
}

/**
 * info getter
 */
void TransCity::get_info(GroupHandle group, const char *name, void *void_buffer) {
    int   *int_buffer   = (int *)void_buffer;
    float *float_buffer = (float *)void_buffer;
    bool  *bool_buffer  = (bool *)void_buffer;

    if (strequ(name, "id")) {
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int i = 0; i < agent_size; i++) {
            int_buffer[i] = agents[i]->get_id();
        }
    } else if (strequ(name, "num")) {
        if (group == 0) { // car
            int_buffer[0] = (int) agents.size();
        } else if (group == -1) { // wall
            int_buffer[0] = 0;
        }
    } else if (strequ(name, "alive")) {
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int i = 0; i < agent_size; i++) {
            bool_buffer[i] = !agents[i]->is_dead();
        }
    } else if (strequ(name, "pos")) {
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int i = 0; i < agent_size; i++) {
            int_buffer[2 * i] = agents[i]->get_pos().x;
            int_buffer[2 * i + 1] = agents[i]->get_pos().y;

        }
    } else if (strequ(name, "action_space")) {
        int_buffer[0] = (int)ACT_NUM;
    } else if (strequ(name, "view_space")) {
        int_buffer[0] = view_height;
        int_buffer[1] = view_width;
        int_buffer[2] = CHANNEL_NUM;
    } else if (strequ(name, "feature_space")) {
        int_buffer[0] = get_feature_size_(group);
    } else {
        LOG(FATAL) << "unsupported info name in TransCity::get_info : " << name;
    }
}


/**
 * render
 */
void TransCity::render() {
    if (first_render) {
        first_render = false;
        render_generator.gen_config(width, height);
    }
    render_generator.render_a_frame(agents, walls, lights, parks, buildings, roads);
}

void TransCity::render_next_file() {
    render_generator.next_file();
}

/**
 *  private utilities
 */
int TransCity::get_feature_size_(GroupHandle group) {
    // embedding + last_action + last_reward + goal
    return embedding_size + static_cast<int>(ACT_NUM) + 1 + 4 + 9;
}


} // namespace trans_city
} // namespace magent

