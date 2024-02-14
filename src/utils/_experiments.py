import gym
import torch
from src.models.particle import ParticleModel
from tqdm import trange

from ..models.pendulum import PendulumModel


def run_gym_simulation(
    env_name, init_state, controller, steps=200, optim_steps=10, seed=None, render=False
):
    if seed is not None:
        torch.manual_seed(seed)

    # create the aggregating tensors and set them to 'nan' to check if
    # simulation breaks
    sim_dict = {}
    sim_dict["states"] = torch.full(
        (steps + 1, controller.dim_s),
        fill_value=float("nan"),
        dtype=torch.float,
    )
    sim_dict["actions"] = torch.full(
        (steps, controller.dim_a),
        fill_value=float("nan"),
        dtype=torch.float,
    )
    sim_dict["costs"] = torch.full(
        (steps, 1), fill_value=float("nan"), dtype=torch.float
    )
    if hasattr(controller, "n_pol"):
        pol_shape = [steps, controller.n_pol]
    else:
        pol_shape = [steps]
    sim_dict["policies"] = torch.full(
        pol_shape + [controller.hz_len, controller.dim_a],
        fill_value=float("nan"),
        dtype=torch.float,
    )
    sim_dict["trace"] = []
    # main iteration loop
    if env_name.lower() == "pendulum":
        model = PendulumModel()
        sim_env = gym.make("Pendulum-v0")
        sim_env.reset()
        # sim_env.unwrapped.l = experiment_params[i]["length"]
        # sim_env.unwrapped.m = experiment_params[i]["mass"]
        sim_env.unwrapped.state = init_state
    else:
        raise ValueError("Invalid environment name: {}.".format(env_name))
    state = init_state
    sim_dict["states"][0] = init_state
    for step in range(steps):
        if render:
            sim_env.render()

        a_seq, trace = controller.forward(
            state, model, params_dist=None, steps=optim_steps
        )
        action = a_seq[0]
        _, _, done, _ = sim_env.step(action)
        state = torch.as_tensor(sim_env.state, dtype=torch.float).unsqueeze(0)
        cost = controller.inst_cost_fn(state.view(1, -1))

        if not step % 10:
            print(
                "Step {0}: action taken {1:.2f}, cost {2:.2f}".format(
                    step, float(action), float(cost)
                )
            )
            print(
                "Current state: theta={0[0]}, theta_dot={0[1]}".format(state.squeeze())
            )
        sim_dict["policies"][step] = controller.pol_mean.detach().clone()
        sim_dict["trace"].append(trace)
        sim_dict["actions"][step] = action
        sim_dict["costs"][step] = cost
        sim_dict["states"][step + 1] = state
        if done:
            sim_env.close()
            break

    # End of episode
    sim_env.close()
    return sim_dict


def run_maze_experiment(
    init_state, controller, env_kwargs={}, steps=200, seed=None, render=False
):
    if seed is not None:
        torch.manual_seed(seed)

    # create the aggregating tensors and set them to 'nan' to check if
    # simulation breaks
    sim_dict = {}
    sim_dict["states"] = torch.full(
        (steps + 1, controller.dim_s),
        fill_value=float("nan"),
        dtype=torch.float,
    )
    sim_dict["actions"] = torch.full(
        (steps, controller.dim_a),
        fill_value=float("nan"),
        dtype=torch.float,
    )
    sim_dict["costs"] = torch.full(
        (steps, 1), fill_value=float("nan"), dtype=torch.float
    )
    if hasattr(controller, "n_pol"):
        pol_shape = [steps, controller.n_pol]
    else:
        pol_shape = [steps]
    sim_dict["policies"] = torch.full(
        pol_shape + [controller.hz_len, controller.dim_a],
        fill_value=float("nan"),
        dtype=torch.float,
    )
    sim_dict["trace"] = []
    model = ParticleModel(**env_kwargs)
    system = ParticleModel(**env_kwargs)

    # main iteration loop
    state = torch.as_tensor(init_state)
    sim_dict["states"][0] = init_state
    # Reset state
    dyn_particles = torch.empty((0, mpf_n_part))
    if USE_MPF:
        mpf = copy(base_mpf)
        dynamics_dist = mpf.prior
    else:
        mpf = None
        dynamics_dist = dynamics_prior
    save_path = save_progress(folder_name=base_path.stem + "/ep{}".format(ep))
    iterator = trange(steps)
    for step in iterator:
        if render:
            system.render()
        # if step == STEPS // 4:  # Changes the simulator mass
        #     system.params_dict["mass"] = system.params_dict["mass"].clone() + LOAD
        # Rollout dynamics, get costs
        # ----------------------------
        # selects next action and makes sure the controller plan is
        # the same as svmpc
        a_seq, trace = controller.forward(state, model, params_dist=None, steps=2)
        # fetch trajectories of last iteration
        s_seq = trace[-1]["trajectories"]
        action = a_seq[0]
        state = system.step(state, action.squeeze())
        grad_dyn = torch.zeros(mpf_steps)
        if step >= WARM_UP and mpf is not None:
            # optimize will automatically update `dynamics_dist`
            grad_dyn, _ = mpf.optimize(
                action.squeeze(), state, bw=mpf_bw, n_steps=mpf_steps
            )

        sim_dict["policies"][step] = controller.pol_mean.detach().clone()
        sim_dict["trace"].append(trace)
        sim_dict["actions"][step] = action
        sim_dict["costs"][step] = cost
        sim_dict["states"][step + 1] = state
        if done:
            sim_env.close()
            break
        tau = torch.cat([tau, state.unsqueeze(0)], dim=0)
        actions = torch.cat([actions, action.unsqueeze(0)])
        rollouts = torch.cat([rollouts, s_seq.unsqueeze(0)], dim=0)
        inst_cost = controller.inst_cost_fn(state.view(1, -1))
        costs = torch.cat([costs, inst_cost.unsqueeze(0)], dim=0)
        if USE_MPF:
            dyn_particles = torch.cat(
                [dyn_particles, mpf.x.detach().clone().t()], dim=0
            )

        if system.with_obstacle:
            if system.obst_map.get_collisions(state[:2]):
                print("\nCrashed at step {}".format(step))
                # print("Last particles state:")
                # print(svmpc.theta)
                break
        if (system.target - state).norm() <= 1.0:
            break

        episode_data = {
            "costs": costs,
            "trajectory": tau,
            "actions": actions,
            "rollouts": rollouts,
            "dyn_particles": dyn_particles,
        }
        save_progress(
            folder_name=save_path.relative_to(base_path.parent), data=episode_data
        )
        # wait for all workers to finish before creating video file
        # BackgroundProcessRunner.wait()
        create_video_from_plots(save_path)
