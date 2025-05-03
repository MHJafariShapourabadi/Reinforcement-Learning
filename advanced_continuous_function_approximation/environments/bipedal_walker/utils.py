import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from IPython.display import display, clear_output

def run_and_display_env(env, agent, num_episodes = 1, max_steps=None):
  # Run the episodes and record the videos
  for episode in range(num_episodes):
      state, info = env.reset()  # Reset the environment for a new episode
      done = False
      frame = env.render()
      total_reward = 0
      steps = 0

      # Display the current state
      if env.unwrapped.render_mode == "rgb_array":
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f"Step: {steps + 1}, Reward: {total_reward}")
        display(plt.gcf())
        # clear_output(wait=True)
        # plt.pause(0.5)  # Pause to visualize the steps

      while not done:
          # Policy: Select action
          action = agent.select_action(state, info)
          next_state, reward, terminated, truncated, info = env.step(action)
          done = terminated or truncated
          frame = env.render()

          # Accumulate the reward
          total_reward += reward
          steps += 1
          state = next_state

          # Display the current state
          if env.unwrapped.render_mode == "rgb_array":
            plt.imshow(frame)
            plt.axis('off')
            plt.title(f"Step: {steps + 1}, Reward: {total_reward}")
            display(plt.gcf())
            # clear_output(wait=True)
            # plt.pause(0.5)  # Pause to visualize the steps

          if max_steps is not None and steps + 1 >= max_steps:
            done = True

      print(f"Episode {episode + 1} completed with total reward: {total_reward} and total steps: {steps}")

  # Close the environment
  env.close()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from gymnasium.wrappers import RecordVideo


def run_and_record_env(env, agent, video_dir, num_episodes = 1, max_steps=None):
  # Create the Frozen Lake environment with video recording
  env = RecordVideo(
      env,
      video_folder=video_dir,
      episode_trigger=lambda episode_id: True,  # Record every episode
      disable_logger=True,  # Disable unnecessary logging
  )

  # Run the episodes and record the videos
  for episode in range(num_episodes):
      state, info = env.reset()  # Reset the environment for a new episode
      done = False
      total_reward = 0.
      steps = 0

      while not done:
          # Policy: Select action
          action = agent.select_action(state, info)
          next_state, reward, terminated, truncated, info = env.step(action)
          done = terminated or truncated

          # Accumulate the reward
          total_reward += reward
          steps += 1
          state = next_state

          if max_steps is not None and steps + 1 >= max_steps:
            done = True

      print(f"Episode {episode + 1} completed with total reward: {total_reward : .3f} and total steps: {steps}")

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

def plot_with_matplotlib(episode_rewards, episode_steps):
    # Plot episode rewards
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label="Episode Rewards", color='b')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards Over Time")
    plt.grid(True)
    plt.legend()

    # Plot episode steps
    plt.subplot(1, 2, 2)
    plt.plot(episode_steps, label="Episode Steps", color='g')
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps Per Episode Over Time")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plot_with_seaborn(episode_rewards, episode_steps):
    # Create a DataFrame for easier plotting
    data = pd.DataFrame({
        "Episode": range(1, len(episode_rewards) + 1),
        "Rewards": episode_rewards,
        "Steps": episode_steps
    })

    # Plot episode rewards
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=data, x="Episode", y="Rewards", color='b', label="Episode Rewards")
    plt.title("Episode Rewards Over Time")
    plt.grid(True)

    # Plot episode steps
    plt.subplot(1, 2, 2)
    sns.lineplot(data=data, x="Episode", y="Steps", color='g', label="Episode Steps")
    plt.title("Steps Per Episode Over Time")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

def plot_q_values_map(qtable, map_size, agent_name, time, savefig_folder=None):
    """Plot the policy learned with Q-values and directions."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the policy
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax,
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.2,
        linecolor="white",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "x-large"},
    ).set(title=f"Learned Q-values for {agent_name} agent\nArrows represent best action\n start state value: {qtable_val_max[0,0]:.2f}  time elapsed: {time:0.2f} sec")
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.2)
        spine.set_color("white")

    if savefig_folder is not None:
        img_title = f"frozenlake_q_values_{map_size}x{map_size}_for_{agent_name}_agent.png"
        fig.savefig(savefig_folder / img_title, bbox_inches="tight")
        
    plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
