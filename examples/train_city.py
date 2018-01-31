"""
Simulating a city with many cars. Every car has a goal.
"""

import argparse
import logging as log
import random
import time
import copy

import numpy as np

import magent

from magent.builtin.tf_model import DeepQNetwork, DeepRecurrentQNetwork
from magent.builtin.rule_model import RandomActor

random.seed(0)
np.random.seed(0)


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"view_width": 9, "view_height": 9})
    cfg.set({"embedding_size": 10})
    cfg.set({"reward_scale": 2, 'ban_penalty': -0.3})

    return cfg


def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents and vertical lines"""
    agent_dense = 0.1
    num_park = 18

    building_min_height = 8
    building_max_height = 50
    building_min_width = 8
    building_max_width = 50
    major_road_height = 8
    major_road_width = 8
    minor_road_height = 4
    minor_road_width = 4

    major_road_wnum = 2
    major_road_hnum = 2

    margin_width = 2
    margin_height = 2

    margin_width += 1
    margin_height += 1

    max_merge_failure_times = 100

    building_pos = []
    light_pos = []

    map_width = map_size - margin_width * 2
    map_height = map_size - margin_height * 2

    assert \
        (map_width - 2 * minor_road_width - major_road_wnum * major_road_width) % (major_road_wnum + 1) == 0, \
        (map_width - 2 * minor_road_width - major_road_wnum * major_road_width) % (major_road_wnum + 1)
    assert \
        (map_height - 2 * minor_road_height - major_road_hnum * major_road_height) % (major_road_hnum + 1) == 0, \
        (map_height - 2 * minor_road_height - major_road_hnum * major_road_height) % (major_road_hnum + 1)

    block_width = (map_width - 2 * minor_road_width - major_road_wnum * major_road_width) / (major_road_wnum + 1)
    block_height = (map_height - 2 * minor_road_height - major_road_hnum * major_road_height) / (major_road_hnum + 1)

    building_width = None
    print(block_width)
    for width in range(building_min_width, building_max_width + 1):
        if (block_width + minor_road_width) % (width + minor_road_width) == 0:
            building_width = width
            break
    assert building_width is not None

    building_height = None
    for height in range(building_min_height, building_max_height + 1):
        if (block_height + minor_road_height) % (height + minor_road_height) == 0:
            building_height = height
            break
    assert building_height is not None

    block_building_wnum = (block_width + minor_road_width) // (building_width + minor_road_width)
    block_building_hnum = (block_height + minor_road_height) // (building_height + minor_road_height)

    region_buildings = []
    for i in range(major_road_wnum + 1):
        tmp_buildings = []
        for j in range(major_road_hnum + 1):
            block_x = margin_width + minor_road_width + i * (block_width + major_road_width)
            block_y = margin_height + minor_road_height + j * (block_height + major_road_height)
            current_buildings = []
            for k in range(block_building_wnum):
                for l in range(block_building_hnum):
                    current_buildings.append((k, l))

            counter = 0
            current_available_buildings = []
            while current_buildings:
                x, y = current_buildings[np.random.choice(len(current_buildings))]
                w = np.random.randint(0, block_building_wnum - x)
                h = np.random.randint(0, block_building_hnum - y)
                if w == 0 and h == 0 and counter <= max_merge_failure_times:
                    counter += 1
                    continue
                checked = True
                for k in range(w + 1):
                    for l in range(h + 1):
                        if (x + k, y + l) not in current_buildings:
                            checked = False
                if checked:
                    building_x = block_x + x * (building_width + minor_road_width)
                    building_y = block_y + y * (building_height + minor_road_height)
                    for k in range(w + 1):
                        for l in range(h + 1):
                            current_buildings.remove((x + k, y + l))
                    current_available_buildings.append((
                        building_x, building_y,
                        (w + 1) * building_width + w * minor_road_width,
                        (h + 1) * building_height + h * minor_road_height))

            building_pos += current_available_buildings
            tmp_buildings.append(current_available_buildings)

            for k in range(block_building_wnum - 1):
                for l in range(block_building_hnum - 1):
                    sx = block_x + k * (building_width + minor_road_width) + building_width - 1
                    sy = block_y + l * (building_height + minor_road_height) + building_height - 1
                    ex = sx + minor_road_width + 1
                    ey = sy + minor_road_height + 1
                    checked = [True] * 4
                    for x, y, w, h in current_available_buildings:
                        if x + w - 1 == sx and y <= sy and ey <= y + h - 1:
                            checked[3] = False
                        if y + h - 1 == sy and x <= sx and ex <= x + w - 1:
                            checked[0] = False
                        if x == ex and y <= sy and ey <= y + h - 1:
                            checked[1] = False
                        if y == ey and x <= sx and ex <= x + w - 1:
                            checked[2] = False
                        if x <= sx and ex <= x + w - 1 and y <= sy and ey <= y + h - 1:
                            checked[0] = checked[1] = checked[2] = checked[3] = False
                            break
                    if sum(checked) >= 3:
                        mask = 0
                        for it in range(4):
                            mask |= checked[it] << it
                        light_pos.append((sx, sy, minor_road_width + 1, minor_road_height + 1, mask))
                    else:
                        if checked[0] and checked[2]:
                            env.add_roads(method='custom', pos=[[sx + 1, sy + 1, ex - sx - 1, ey - sy - 1, 1]])
                        elif checked[1] and checked[3]:
                            env.add_roads(method='custom', pos=[[sx + 1, sy + 1, ex - sx - 1, ey - sy - 1, 0]])
            for k in range(block_building_wnum):
                for l in range(block_building_hnum):
                    sx = block_x + k * (building_width + minor_road_width) + building_width
                    sy = block_y + l * (building_height + minor_road_height)
                    ex = sx + minor_road_width - 1
                    ey = sy + building_height - 1
                    if ex <= block_x + block_width - 1 and ey <= block_y + block_height - 1:
                        checked = True
                        for x, y, w, h in current_available_buildings:
                            if x <= sx and ex <= x + w - 1 and y <= sy and ey <= y + h - 1:
                                checked = False
                                break
                        if checked:
                            env.add_roads(method='custom', pos=[[sx, sy, minor_road_width, building_height, 1]])
                    sx = block_x + k * (building_width + minor_road_width)
                    sy = block_y + l * (building_height + minor_road_height) + building_height
                    ex = sx + building_width - 1
                    ey = sy + minor_road_height - 1
                    if ex <= block_x + block_width - 1 and ey <= block_y + block_height - 1:
                        checked = True
                        for x, y, w, h in current_available_buildings:
                            if x <= sx and ex <= x + w - 1 and y <= sy and ey <= y + h - 1:
                                checked = False
                                break
                        if checked:
                            env.add_roads(method='custom', pos=[[sx, sy, building_width, minor_road_height, 0]])
        region_buildings.append(tmp_buildings)

    for i in range(major_road_wnum + 1):
        for j in range(major_road_hnum + 1):
            block_x = margin_width + minor_road_width + i * (block_width + major_road_width)
            block_y = margin_height + minor_road_height + j * (block_height + major_road_height)
            if i < major_road_wnum:
                for k in range(block_building_hnum):
                    if k < block_building_hnum - 1:
                        sx = block_x + block_width - 1
                        sy = block_y + k * (building_height + minor_road_height) + building_height - 1
                        ex = sx + major_road_width + 1
                        ey = sy + minor_road_width + 1
                        checked = [True] * 4
                        for x, y, w, h in region_buildings[i][j]:
                            if sx == x + w - 1 and y <= sy and ey <= y + h - 1:
                                checked[3] = False
                        for x, y, w, h in region_buildings[i + 1][j]:
                            if ex == x and y <= sy and ey <= y + h - 1:
                                checked[1] = False
                        if sum(checked) >= 3:
                            mask = 0
                            for it in range(4):
                                mask |= checked[it] << it
                            light_pos.append((sx, sy, major_road_width + 1, minor_road_height + 1, mask))
                        else:
                            env.add_roads(method='custom', pos=[[sx + 1, sy + 1, major_road_width, minor_road_height, 0]])

                    sx = block_x + block_width
                    sy = block_y + k * (building_height + minor_road_height)
                    env.add_roads(method='custom', pos=[[sx, sy, major_road_width, building_height, 1]])
            if j < major_road_hnum:
                for k in range(block_building_wnum):
                    if k < block_building_wnum - 1:
                        sx = block_x + k * (building_width + minor_road_width) + building_width - 1
                        sy = block_y + block_height - 1
                        ex = sx + minor_road_width + 1
                        ey = sy + major_road_height + 1
                        checked = [True] * 4
                        for x, y, w, h in region_buildings[i][j]:
                            if sy == y + h - 1 and x <= sx and ex <= x + w - 1:
                                checked[0] = False
                        for x, y, w, h in region_buildings[i][j + 1]:
                            if ey == y and x <= sx and ex <= x + w - 1:
                                checked[2] = False
                        if sum(checked) >= 3:
                            mask = 0
                            for it in range(4):
                                mask |= checked[it] << it
                            light_pos.append((sx, sy, minor_road_width + 1, major_road_height + 1, mask))
                        else:
                            env.add_roads(method='custom', pos=[[sx + 1, sy + 1, minor_road_width, major_road_height, 0]])

                    sx = block_x + k * (building_width + minor_road_width)
                    sy = block_y + block_height
                    env.add_roads(method='custom', pos=[[sx, sy, building_width, major_road_height, 0]])
            if i < major_road_hnum and j < major_road_hnum:
                light_pos.append((
                    margin_width + minor_road_width + block_width * (i + 1) + major_road_width * i - 1,
                    margin_height + minor_road_height + block_height * (j + 1) + major_road_height * j - 1,
                    major_road_width + 1,
                    major_road_height + 1,
                    15
                ))

    filled = set()
    for pos in building_pos + light_pos:
        x0, y0, w, h = pos[:4]
        for x in range(x0, x0 + w):
            for y in range(y0, y0 + h):
                filled.add((x, y))

    building_pos = np.array(building_pos)
    env.add_buildings(method='custom', pos=building_pos)
    index = np.arange(len(building_pos))
    env.add_parks(method="custom", pos=building_pos[np.random.choice(index, num_park)])
    env.add_traffic_lights(method='custom', pos=light_pos)

    n = map_size * map_size * agent_dense

    env.add_agents(method="random", n=n)

    return filled, n


def get_enter_pos(map_size, filled, n, width=8):
    ret = []
    i = 0

    enters = []
    for x in range(1, map_size - 1):
        for y in range(1, width):
            enters.append([x, y])
            enters.append([y, x])
        for y in range(map_size - width - 1, map_size - 1):
            enters.append([x, y])
            enters.append([y, x])

    while i < n:
        x, y = enters[np.random.choice(range(len(enters)))]
        while (x, y) in filled:
            x, y = enters[np.random.choice(range(len(enters)))]

        ret.append([x, y])
        i += 1
    return ret


def play_a_round(env, map_size, handles, models, print_every, train=True, render=False, eps=None):
    env.reset()
    filled, max_car_num = generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n = len(handles)
    obs = [[] for _ in range(n)]
    ids = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    sample_buffer = magent.utility.EpisodesBuffer(capacity=1000)
    total_reward = [0 for _ in range(n)]

    print("===== sample =====")
    print("eps %.2f number %s" % (eps, nums))
    start_time = time.time()
    while not done:
        # stat info
        begin_nums = [env.get_num(handle) for handle in handles]

        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(handles[i])
            ids[i] = env.get_agent_id(handles[i])
            acts[i] = models[i].infer_action(obs[i], ids[i], 'e_greedy', eps=eps)
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        # sample
        step_reward = []
        for i in range(n):
            rewards = env.get_reward(handles[i])
            if train:
                alives = env.get_alive(handles[i])
                sample_buffer.record_step(ids[i], obs[i], acts[i], rewards, alives)
            s = sum(rewards)
            step_reward.append(s)
            total_reward[i] += s

        # render
        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        # add new cars
        if nums[0] < max_car_num and False:
            env.add_agents(method="custom", pos=get_enter_pos(map_size, filled,
                                                              max_car_num - nums[0]))

        if step_ct % print_every == 0:
            print("step %3d,  nums: %s reward: %s,  total_reward: %s  enter: %d" %
                  (step_ct, nums, np.around(step_reward, 2), np.around(total_reward, 2),
                   sum(begin_nums) - sum(nums)))
        step_ct += 1
        if step_ct > 400:
            break

    sample_time = time.time() - start_time
    print("steps: %d,  total time: %.2f,  step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    # train
    total_loss, value = 0, 0
    if train:
        print("===== train =====")
        start_time = time.time()
        total_loss, value = models[0].train(sample_buffer, 500)
        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    def round_list(l):
        return [round(x, 2) for x in l]

    return total_loss, nums, round_list(total_reward), value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=198)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--name", type=str, default="city")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('--alg', default='dqn', choices=['dqn', 'drqn', 'a2c'])
    args = parser.parse_args()

    # set logger
    magent.utility.init_logger(args.name)

    # init the game
    env = magent.TransCity(get_config(args.map_size))
    env.set_render_dir("build/render")

    # init models
    batch_size = 256
    target_update = 2000
    train_freq = 5

    handles = [0]

    models = []
    models.append(DeepQNetwork(env, handles[0], "cars",
                               batch_size=batch_size,
                               memory_size=2 ** 21, target_update=target_update,
                               train_freq=train_freq))

    # load if
    savedir = 'save_model'
    if args.load_from is not None:
        start_from = args.load_from
        print("load ... %d" % start_from)
        for model in models:
            model.load(savedir, start_from)
    else:
        start_from = 0

    # print debug info
    print(args)
    print("view_space", env.get_view_space(handles[0]))
    print("feature_space", env.get_feature_space(handles[0]))

    # play
    start = time.time()
    for k in range(start_from, start_from + args.n_round):
        tic = time.time()
        eps = magent.utility.piecewise_decay(k, [0, 200, 500], [1, 0.2, 0.1]) if not args.greedy else 0
        loss, num, reward, value = play_a_round(env, args.map_size, handles, models,
                                                train=args.train, print_every=50,
                                                render=args.render or (k + 1) % args.render_every == 0,
                                                eps=eps)  # for e-greedy

        log.info("round %d\t loss: %s\t num: %s\t reward: %s\t value: %s" % (k, loss, num, reward, value))
        print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

        # save models
        if (k + 1) % args.save_every == 0 and args.train:
            print("save model... ")
            for model in models:
                model.save(savedir, k)
