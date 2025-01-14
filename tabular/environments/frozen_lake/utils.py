import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def postprocess(n_runs, episodes, rewards, steps, map_size, name):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=n_runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten(),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])
    res["name"] = np.repeat(f"{name}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    st["name"] = np.repeat(f"{name}", st.shape[0])
    
    return res, st


def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if np.abs(qtable_val_max.flatten()[idx]) > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, env, map_size, agent_name, time, savefig_folder):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title(f"Last frame for {agent_name} agent")

    ##################################################
    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.2,
        linecolor="white",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "x-large"},
    ).set(title=f"Learned Q-values for {agent_name} agent\nArrows represent best action\n start state value: {qtable_val_max[0,0]:.2f}  time elapsed: {time:0.2f} sec")

    ##################################################
    # # Plot the policy
    # im = ax[1].imshow(qtable_val_max , cmap="Blues")
    # ax[1].set_title(f"Learned Q-values for {agent_name} agent\nArrows represent best action\n start state value: {qtable_val_max[0,0]:.2f}  time elapsed: {time:0.2f} sec")
    # ax[1].axis("off")
    # cbar = ax[1].figure.colorbar(im, ax=ax[1])
    # for i in range(qtable_directions.shape[0]):
    #     for j in range(qtable_directions.shape[1]):
    #         text = ax[1].text(j, i, qtable_directions[i, j],
    #                    ha="center", va="center", color="w")
    ##################################################

    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.2)
        spine.set_color("white")
    img_title = f"frozenlake_q_values_{map_size}x{map_size}_for_{agent_name}_agent.png"
    fig.savefig(savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def plot_states_actions_distribution(states, actions, map_size, agent_name, savefig_folder):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title(f"States for {agent_name}")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title(f"Actions for {agent_name}")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}_for_{agent_name}_agent.png"
    fig.savefig(savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def plot_steps_and_rewards(rewards_df, steps_df, savefig_folder):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        # data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0]
        data=rewards_df, x="Episodes", y="cum_rewards", hue="name", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    # sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="name", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        # axi.legend(title="map size")
        axi.legend(title="name")
    fig.tight_layout()
    img_title = "frozenlake_steps_and_rewards.png"
    fig.savefig(savefig_folder / img_title, bbox_inches="tight")
    plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


def run_and_record_env(env, agent, explorer, video_dir, num_episodes = 1):
  # Create the Frozen Lake environment with video recording
  env = RecordVideo(
      env,
      video_folder=video_dir,
      episode_trigger=lambda episode_id: True,  # Record every episode
      disable_logger=True,  # Disable unnecessary logging
  )

  # Run the episodes and record the videos
  for episode in range(num_episodes):
      state, _ = env.reset()  # Reset the environment for a new episode
      done = False
      total_reward = 0

      while not done:
        #   Random policy: Select a random action
        #   action = env.action_space.sample()
        #   action = explorer(env, state)
            action = explorer.choose_action(env.action_space, state, agent.qtable)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        #   Accumulate the reward
            total_reward += reward
            state = next_state

      print(f"Episode {episode + 1} completed with total reward: {total_reward}")

  # Close the environment
  env.close()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import subprocess
import platform

def play_videos(video_dir):
    # Play the recorded videos using the default media player
    print("Playing back the recorded episodes:")
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    for video_file in sorted(video_files):  # Play videos in order
        video_path = os.path.join(video_dir, video_file)
        print(f"Playing {video_file}")

        # Use default media player to open the video
        if platform.system() == "Windows":
            os.startfile(video_path)  # For Windows
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", video_path])
        else:  # Linux/Unix
            subprocess.run(["xdg-open", video_path])

    print("All videos have been played.")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
from IPython.display import Video, display

def play_videos_in_jupyter(video_dir):
  # Play the recorded videos in Jupyter Notebook
  print("Playing back the recorded episodes:")
  video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

  for video_file in sorted(video_files):  # Play videos in order
      video_path = os.path.join(video_dir, video_file)
      print(f"Playing {video_file}")
      display(Video(video_path, embed=True))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os

def remove_videos(video_dir):
  # Remove all video files in the directory
  if os.path.exists(video_dir):
      video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
      for video_file in video_files:
          file_path = os.path.join(video_dir, video_file)
          os.remove(file_path)
          print(f"Deleted: {file_path}")
      print("All video files have been deleted.")
  else:
      print(f"Directory '{video_dir}' does not exist.")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%