import wandb
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
import io
from PIL import Image
from jaxgcrl.agents.crl.losses import energy_fn

def visualize_goals_2d(start_xy, proposed_goals_xy, 
                       last_traj_states_xy, intermediate_traj_states_xy, wandb_key,
                       x_bounds=None, y_bounds=None):
    '''Visualize 2D goals and trajectories with interactive Plotly (simplified version).
    - start_xy: (num_samples, 2) array of start states
    - proposed_goals_xy: (num_samples, 2) array of proposed goals
    - last_traj_states_xy: (num_samples, 2) array of last trajectory states
    - intermediate_traj_states_xy: (num_samples, num_intermediate_states, 2) array of intermediate trajectory states
    - wandb_key: str, key to log the plot in WandB
    - x_bounds: tuple (min, max) for x-axis range, or None for auto
    - y_bounds: tuple (min, max) for y-axis range, or None for auto
    '''
    assert start_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert proposed_goals_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert last_traj_states_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert intermediate_traj_states_xy.shape[2] == 2, "Goal visualization only supported for 2D goals"
    
    fig = go.Figure()
    
    num_samples = start_xy.shape[0]
    
    # Plot trajectories and arrows first (so points appear on top)
    for i in range(num_samples):
        # Intermediate trajectory states
        fig.add_trace(go.Scatter(
            x=intermediate_traj_states_xy[i, :, 0],
            y=intermediate_traj_states_xy[i, :, 1],
            mode='markers',
            marker=dict(color='purple', size=2, opacity=0.4),
            showlegend=(i == 0),
            name='Trajectory Points' if i == 0 else '',
            hovertemplate='Intermediate<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
        ))
        
        # Full trajectory line
        full_traj_xy = np.vstack([
            start_xy[i:i+1],
            intermediate_traj_states_xy[i],
            last_traj_states_xy[i:i+1]
        ])
        
        fig.add_trace(go.Scatter(
            x=full_traj_xy[:, 0],
            y=full_traj_xy[:, 1],
            mode='lines',
            line=dict(color='purple', width=1),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Dashed line from proposed goal to last trajectory state
        fig.add_trace(go.Scatter(
            x=[proposed_goals_xy[i, 0], last_traj_states_xy[i, 0]],
            y=[proposed_goals_xy[i, 1], last_traj_states_xy[i, 1]],
            mode='lines',
            line=dict(color='orange', width=1.5, dash='dash'),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Plot main point clouds
    fig.add_trace(go.Scatter(
        x=start_xy[:, 0],
        y=start_xy[:, 1],
        mode='markers',
        marker=dict(color='blue', size=4, opacity=0.6),
        name='Start States',
        hovertemplate='Start State<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=proposed_goals_xy[:, 0],
        y=proposed_goals_xy[:, 1],
        mode='markers',
        marker=dict(color='orange', size=4, opacity=0.6),
        name='Proposed Goals',
        hovertemplate='Proposed Goal<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=last_traj_states_xy[:, 0],
        y=last_traj_states_xy[:, 1],
        mode='markers',
        marker=dict(color='green', size=4, opacity=0.6),
        name='Reached Goal',
        hovertemplate='Reached Goal<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    # Configure axis settings based on whether bounds are provided
    xaxis_config = dict(scaleanchor="y", scaleratio=1, constrain='domain')
    yaxis_config = dict(constrain='domain')
    
    if x_bounds is not None:
        xaxis_config['range'] = list(x_bounds)
    
    if y_bounds is not None:
        yaxis_config['range'] = list(y_bounds)
    
    # Update layout
    fig.update_layout(
        title="Agent Trajectories and Goal Proposals",
        xaxis_title="x",
        yaxis_title="y",
        width=2100,
        height=2100,
        hovermode='closest',
        showlegend=True,
        xaxis=xaxis_config,
        yaxis=yaxis_config
    )
    
    # Log to WandB as interactive plot
    wandb.log({wandb_key: fig})


def visualize_dual_crl_trajectories_2d(start_xy, gc_final_xy, ep_final_xy, gc_proposed_goals_xy, ep_proposed_goals_xy,
                                       gc_intermediate_xy_list, ep_intermediate_xy_list, wandb_key,
                                       x_bounds=None, y_bounds=None):
    '''Visualize 2D trajectories for dual CRL with GC and EP phases in a 2x2 grid.
    - start_xy: (num_samples, 2) array of start states (should be 4 trajectories)
    - gc_final_xy: (num_samples, 2) array of final states from GC rollout
    - ep_final_xy: (num_samples, 2) array of final states from EP rollout
    - gc_proposed_goals_xy: (num_samples, 2) array of GC proposed goals
    - ep_proposed_goals_xy: (num_samples, 2) array of EP proposed goals
    - gc_intermediate_xy_list: list of (num_gc_intermediate, 2) arrays for GC intermediate states
    - ep_intermediate_xy_list: list of (num_ep_intermediate, 2) arrays for EP intermediate states
    - wandb_key: str, key to log the plot in WandB
    - x_bounds: tuple (min, max) for x-axis range, or None for auto
    - y_bounds: tuple (min, max) for y-axis range, or None for auto
    '''
    assert start_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert gc_final_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert ep_final_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert gc_proposed_goals_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert ep_proposed_goals_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert len(gc_intermediate_xy_list) == len(ep_intermediate_xy_list) == start_xy.shape[0], "Mismatch in number of trajectories"
    
    num_samples = start_xy.shape[0]
    num_trajectories = min(4, num_samples)  # Plot up to 4 trajectories
    
    # Create 2x2 subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Trajectory 1', 'Trajectory 2', 'Trajectory 3', 'Trajectory 4'),
        horizontal_spacing=0.15,
        vertical_spacing=0.15,
        subplot_titles_font_size=14
    )
    
    # Helper function to plot a single trajectory in a subplot
    def plot_trajectory_in_subplot(i, row, col):
        """Plot trajectory i in subplot at (row, col)"""
        # Get GC and EP intermediate states for this trajectory
        gc_intermediate = np.array(gc_intermediate_xy_list[i])  # (num_gc_intermediate, 2)
        ep_intermediate = np.array(ep_intermediate_xy_list[i])  # (num_ep_intermediate, 2)
        
        # Ensure all arrays are 2D with shape (N, 2) - extract single row and ensure 2D
        start_point = start_xy[i:i+1] if start_xy.ndim == 2 else np.array([start_xy[i]])  # (1, 2)
        gc_final_point = gc_final_xy[i:i+1] if gc_final_xy.ndim == 2 else np.array([gc_final_xy[i]])  # (1, 2)
        ep_final_point = ep_final_xy[i:i+1] if ep_final_xy.ndim == 2 else np.array([ep_final_xy[i]])  # (1, 2)
        
        # Ensure intermediate arrays are 2D
        if gc_intermediate.ndim == 1:
            gc_intermediate = gc_intermediate.reshape(1, -1)
        if ep_intermediate.ndim == 1:
            ep_intermediate = ep_intermediate.reshape(1, -1)
        
        # Ensure all points have shape (N, 2) before vstacking
        start_point = start_point.reshape(-1, 2)
        gc_final_point = gc_final_point.reshape(-1, 2)
        ep_final_point = ep_final_point.reshape(-1, 2)
        if len(gc_intermediate) > 0:
            gc_intermediate = gc_intermediate.reshape(-1, 2)
        if len(ep_intermediate) > 0:
            ep_intermediate = ep_intermediate.reshape(-1, 2)
        
        # Full trajectory line: start → GC intermediate → GC final → EP intermediate → EP final
        full_traj_points = [start_point]
        if len(gc_intermediate) > 0:
            full_traj_points.append(gc_intermediate)
        full_traj_points.append(gc_final_point)
        if len(ep_intermediate) > 0:
            full_traj_points.append(ep_intermediate)
        full_traj_points.append(ep_final_point)
        full_traj_xy = np.vstack(full_traj_points)
        
        # Main trajectory line connecting all points
        fig.add_trace(go.Scatter(
            x=full_traj_xy[:, 0],
            y=full_traj_xy[:, 1],
            mode='lines',
            line=dict(color='blue', width=2),
            opacity=0.6,
            showlegend=(i == 0),
            name='Full Trajectory' if i == 0 else '',
            hoverinfo='skip'
        ), row=row, col=col)
        
        # Line from GC final state to GC proposed goal
        fig.add_trace(go.Scatter(
            x=[gc_final_xy[i, 0], gc_proposed_goals_xy[i, 0]],
            y=[gc_final_xy[i, 1], gc_proposed_goals_xy[i, 1]],
            mode='lines',
            line=dict(color='orange', width=1.5, dash='dash'),
            opacity=0.6,
            showlegend=(i == 0),
            name='GC Goal' if i == 0 else '',
            hoverinfo='skip'
        ), row=row, col=col)
        
        # Line from EP final state to EP proposed goal
        fig.add_trace(go.Scatter(
            x=[ep_final_xy[i, 0], ep_proposed_goals_xy[i, 0]],
            y=[ep_final_xy[i, 1], ep_proposed_goals_xy[i, 1]],
            mode='lines',
            line=dict(color='red', width=1.5, dash='dash'),
            opacity=0.6,
            showlegend=(i == 0),
            name='EP Goal' if i == 0 else '',
            hoverinfo='skip'
        ), row=row, col=col)
        
        # GC intermediate states as markers
        if len(gc_intermediate) > 0:
            fig.add_trace(go.Scatter(
                x=gc_intermediate[:, 0],
                y=gc_intermediate[:, 1],
                mode='markers',
                marker=dict(color='lightblue', size=4, opacity=0.7),
                showlegend=(i == 0),
                name='GC Intermediate' if i == 0 else '',
                hovertemplate='GC Intermediate<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
            ), row=row, col=col)
        
        # EP intermediate states as markers
        if len(ep_intermediate) > 0:
            fig.add_trace(go.Scatter(
                x=ep_intermediate[:, 0],
                y=ep_intermediate[:, 1],
                mode='markers',
                marker=dict(color='pink', size=4, opacity=0.7),
                showlegend=(i == 0),
                name='EP Intermediate' if i == 0 else '',
                hovertemplate='EP Intermediate<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
            ), row=row, col=col)
        
        # Plot main points
        fig.add_trace(go.Scatter(
            x=[start_xy[i, 0]],
            y=[start_xy[i, 1]],
            mode='markers',
            marker=dict(color='blue', size=8, opacity=0.8),
            showlegend=(i == 0),
            name='Start State' if i == 0 else '',
            hovertemplate='Start State<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=[gc_final_xy[i, 0]],
            y=[gc_final_xy[i, 1]],
            mode='markers',
            marker=dict(color='green', size=8, opacity=0.8),
            showlegend=(i == 0),
            name='GC Final' if i == 0 else '',
            hovertemplate='GC Final<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=[ep_final_xy[i, 0]],
            y=[ep_final_xy[i, 1]],
            mode='markers',
            marker=dict(color='purple', size=8, opacity=0.8),
            showlegend=(i == 0),
            name='EP Final' if i == 0 else '',
            hovertemplate='EP Final<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=[gc_proposed_goals_xy[i, 0]],
            y=[gc_proposed_goals_xy[i, 1]],
            mode='markers',
            marker=dict(color='orange', size=8, opacity=0.8, symbol='star'),
            showlegend=(i == 0),
            name='GC Goal' if i == 0 else '',
            hovertemplate='GC Goal<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=[ep_proposed_goals_xy[i, 0]],
            y=[ep_proposed_goals_xy[i, 1]],
            mode='markers',
            marker=dict(color='red', size=8, opacity=0.8, symbol='star'),
            showlegend=(i == 0),
            name='EP Goal' if i == 0 else '',
            hovertemplate='EP Goal<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
        ), row=row, col=col)
    
    # Plot up to 4 trajectories in 2x2 grid
    subplot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for i in range(num_trajectories):
        row, col = subplot_positions[i]
        plot_trajectory_in_subplot(i, row, col)
    
    # Configure axis settings for all subplots
    for row in range(1, 3):
        for col in range(1, 3):
            axis_config = {}
            if x_bounds is not None:
                axis_config['range'] = list(x_bounds)
            if y_bounds is not None:
                axis_config['range'] = list(y_bounds)
            
            fig.update_xaxes(axis_config, row=row, col=col)
            fig.update_yaxes(axis_config, row=row, col=col)
            fig.update_xaxes(scaleanchor="y", scaleratio=1, row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title="Dual CRL Trajectories: GC and EP Phases (2x2 Grid)",
        width=2800,
        height=2800,
        hovermode='closest',
        showlegend=True,
        title_font_size=18
    )
    
    # Log to WandB as interactive plot
    wandb.log({wandb_key: fig})


def visualize_kde_heatmap(data_xy, plot_title, wandb_key, x_bounds=None, y_bounds=None):
    '''Visualize heatmap of xy data in 2D using seaborn KDE.
    - data_xy: (num_points, 2) array of xy data
    - plot_title: str, title for the plot
    - wandb_key: str, key to log the plot in WandB
    - x_bounds: tuple (min, max) for x-axis range, or None for auto
    - y_bounds: tuple (min, max) for y-axis range, or None for auto
    '''
    assert data_xy.shape[1] == 2, "Heatmap visualization only supported for 2D goals"
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create KDE plot
    sns.kdeplot(
        x=data_xy[:, 0],
        y=data_xy[:, 1],
        fill=True,
        cmap='viridis',
        ax=ax,
        cbar=True
    )
    
    # Set bounds if provided
    if x_bounds is not None:
        ax.set_xlim(x_bounds)
    if y_bounds is not None:
        ax.set_ylim(y_bounds)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{plot_title} Distribution (KDE) for {data_xy.shape[0]} points')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Save to buffer and log to WandB
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    pil_image = Image.open(buf)
    wandb.log({wandb_key: wandb.Image(pil_image)})

def visualize_q_function_2d(actor, sa_encoder, g_encoder, actor_params, critic_params, 
                            state, goal_indices, x_bounds, y_bounds, wandb_key, 
                            energy_fn_name, grid_resolution=50):
    '''Visualize Q-function as a heatmap over 2D goal space with policy-generated actions.
    - actor: actor network
    - sa_encoder: state-action encoder network
    - g_encoder: goal encoder network
    - actor_params: actor network parameters
    - critic_params: critic network parameters
    - state: (state_dim,) array - the state to condition on
    - goal_indices: indices for goal dimensions
    - x_bounds: tuple (min, max) for x-axis range
    - y_bounds: tuple (min, max) for y-axis range
    - wandb_key: str, key to log the plot in WandB
    - energy_fn_name: str, type of energy function ('norm', 'l2', 'dot', 'cosine')
    - grid_resolution: int, number of points per axis
    - key: JAX random key for sampling actions
    '''
    # Create grid of goal positions
    x = np.linspace(x_bounds[0], x_bounds[1], grid_resolution)
    y = np.linspace(y_bounds[0], y_bounds[1], grid_resolution)
    xx, yy = np.meshgrid(x, y)
    
    # Flatten grid for batch processing
    goals_grid = np.stack([xx.flatten(), yy.flatten()], axis=1)  # (grid_resolution^2, 2)
    num_goals = goals_grid.shape[0]
    
    # Create observations by concatenating state with each goal
    state_expanded = np.tile(state, (num_goals, 1))  # (grid_resolution^2, state_dim)
    obs_batch = np.concatenate([state_expanded, goals_grid], axis=1)  # (grid_resolution^2, obs_dim)
    
    # Sample actions from policy for each goal
    means, _ = actor.apply(actor_params, obs_batch)
    actions = jax.nn.tanh(means)  # (grid_resolution^2, action_dim)
    
    # Encode state-action pairs
    sa_pairs = np.concatenate([state_expanded, actions], axis=1)  # (grid_resolution^2, state_dim + action_dim)
    
    # Handle both single critic and ensemble cases
    sa_encoder_params = critic_params['sa_encoder']
    g_encoder_params = critic_params['g_encoder']
    if isinstance(sa_encoder_params, list):
        # Ensemble case: use first critic for visualization
        sa_encoder_params = sa_encoder_params[0]
        g_encoder_params = g_encoder_params[0]
    
    phi_sa = sa_encoder.apply(sa_encoder_params, sa_pairs)  # (grid_resolution^2, repr_dim)
    
    # Encode all goals in batch
    psi_g = g_encoder.apply(g_encoder_params, goals_grid)  # (grid_resolution^2, repr_dim)

    q_values = energy_fn(energy_fn_name, phi_sa, psi_g)
    
    # Reshape back to grid
    q_grid = q_values.reshape(grid_resolution, grid_resolution)
    
    # Use percentile-based clipping to handle outliers better
    # This makes the colormap more informative for the main range
    q_flat = q_grid.flatten()
    vmin = np.percentile(q_flat, 2)  # 2nd percentile
    vmax = np.percentile(q_flat, 98)  # 98th percentile
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(q_grid, extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
                   origin='lower', cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
    
    # Mark the current state position
    state_goal_pos = state[goal_indices]
    ax.plot(state_goal_pos[0], state_goal_pos[1], 'r*', markersize=20, 
            label=f'Current State: ({state_goal_pos[0]:.2f}, {state_goal_pos[1]:.2f})')
    
    ax.set_xlabel('Goal x')
    ax.set_ylabel('Goal y')
    ax.set_title(f'Q-Function Landscape\nState: [{state_goal_pos[0]:.2f}, {state_goal_pos[1]:.2f}]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Q-value', rotation=270, labelpad=20)
    
    # Save to buffer and log to WandB
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    pil_image = Image.open(buf)
    wandb.log({wandb_key: wandb.Image(pil_image)})


def visualize_td3_goals_2d(start_xy, proposed_goals_xy, final_states_xy, wandb_key,
                            intermediate_xy=None, x_bounds=None, y_bounds=None):
    '''Visualize 2D goals for TD3-style goal-conditioned RL.
    Shows trajectories from start state to final achieved state, with proposed goals.
    - start_xy: (num_samples, 2) array of start states
    - proposed_goals_xy: (num_samples, 2) array of proposed goals (from replay buffer)
    - final_states_xy: (num_samples, 2) array of final achieved states
    - wandb_key: str, key to log the plot in WandB
    - intermediate_xy: optional (num_samples, num_intermediate, 2) array of intermediate trajectory states
    - x_bounds: tuple (min, max) for x-axis range, or None for auto
    - y_bounds: tuple (min, max) for y-axis range, or None for auto
    '''
    assert start_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert proposed_goals_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert final_states_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    if intermediate_xy is not None:
        assert intermediate_xy.shape[2] == 2, "Intermediate trajectory visualization only supported for 2D"
    
    fig = go.Figure()
    
    num_samples = start_xy.shape[0]
    
    # Plot trajectories first (so points appear on top)
    for i in range(num_samples):
        # If we have intermediate states, plot full trajectory line through them
        if intermediate_xy is not None:
            # Build full trajectory: start -> intermediate -> final
            full_traj_xy = np.vstack([
                start_xy[i:i+1],
                intermediate_xy[i],
                final_states_xy[i:i+1]
            ])
            
            # Full trajectory line
            fig.add_trace(go.Scatter(
                x=full_traj_xy[:, 0],
                y=full_traj_xy[:, 1],
                mode='lines',
                line=dict(color='purple', width=1.5),
                opacity=0.4,
                showlegend=(i == 0),
                name='Trajectory' if i == 0 else '',
                hoverinfo='skip'
            ))
            
            # Intermediate trajectory points
            fig.add_trace(go.Scatter(
                x=intermediate_xy[i, :, 0],
                y=intermediate_xy[i, :, 1],
                mode='markers',
                marker=dict(color='purple', size=3, opacity=0.5),
                showlegend=(i == 0),
                name='Trajectory Points' if i == 0 else '',
                hovertemplate='Intermediate<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
            ))
        else:
            # Simple line from start state to final state
            fig.add_trace(go.Scatter(
                x=[start_xy[i, 0], final_states_xy[i, 0]],
                y=[start_xy[i, 1], final_states_xy[i, 1]],
                mode='lines',
                line=dict(color='purple', width=1.5),
                opacity=0.4,
                showlegend=(i == 0),
                name='Trajectory' if i == 0 else '',
                hoverinfo='skip'
            ))
        
        # Dashed line from proposed goal to final state (goal-achievement gap)
        fig.add_trace(go.Scatter(
            x=[proposed_goals_xy[i, 0], final_states_xy[i, 0]],
            y=[proposed_goals_xy[i, 1], final_states_xy[i, 1]],
            mode='lines',
            line=dict(color='orange', width=1, dash='dash'),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Plot main point clouds
    fig.add_trace(go.Scatter(
        x=start_xy[:, 0],
        y=start_xy[:, 1],
        mode='markers',
        marker=dict(color='blue', size=6, opacity=0.7),
        name='Start States',
        hovertemplate='Start State<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=proposed_goals_xy[:, 0],
        y=proposed_goals_xy[:, 1],
        mode='markers',
        marker=dict(color='orange', size=6, opacity=0.7),
        name='Proposed Goals',
        hovertemplate='Proposed Goal<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=final_states_xy[:, 0],
        y=final_states_xy[:, 1],
        mode='markers',
        marker=dict(color='green', size=6, opacity=0.7),
        name='Achieved States',
        hovertemplate='Achieved State<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    # Configure axis settings based on whether bounds are provided
    xaxis_config = dict(scaleanchor="y", scaleratio=1, constrain='domain')
    yaxis_config = dict(constrain='domain')
    
    if x_bounds is not None:
        xaxis_config['range'] = list(x_bounds)
    
    if y_bounds is not None:
        yaxis_config['range'] = list(y_bounds)
    
    # Update layout
    fig.update_layout(
        title="TD3 Goal Proposals and Achievements",
        xaxis_title="x",
        yaxis_title="y",
        width=2100,
        height=2100,
        hovermode='closest',
        showlegend=True,
        xaxis=xaxis_config,
        yaxis=yaxis_config
    )
    
    # Log to WandB as interactive plot
    wandb.log({wandb_key: fig})


def visualize_td3_q_function_2d(policy_network, q_network, normalizer_params, policy_params, q_params,
                                state, goal_indices, x_bounds, y_bounds, wandb_key, 
                                grid_resolution=100):
    '''Visualize TD3 Q-function as a heatmap over 2D goal space with policy-generated actions.
    - policy_network: TD3 policy network
    - q_network: TD3 Q network
    - normalizer_params: normalizer parameters
    - policy_params: policy network parameters
    - q_params: Q network parameters
    - state: (state_dim,) array - the state to condition on
    - goal_indices: indices for goal dimensions
    - x_bounds: tuple (min, max) for x-axis range
    - y_bounds: tuple (min, max) for y-axis range
    - wandb_key: str, key to log the plot in WandB
    - grid_resolution: int, number of points per axis
    '''
    # Create grid of goal positions
    x = np.linspace(x_bounds[0], x_bounds[1], grid_resolution)
    y = np.linspace(y_bounds[0], y_bounds[1], grid_resolution)
    xx, yy = np.meshgrid(x, y)
    
    # Flatten grid for batch processing
    goals_grid = np.stack([xx.flatten(), yy.flatten()], axis=1)  # (grid_resolution^2, 2)
    num_goals = goals_grid.shape[0]
    
    # Create observations by concatenating state with each goal
    state_expanded = np.tile(state, (num_goals, 1))  # (grid_resolution^2, state_dim)
    obs_batch = jnp.concatenate([state_expanded, goals_grid], axis=1)  # (grid_resolution^2, obs_dim)
    
    # Sample actions from policy for each goal
    actions = policy_network.apply(normalizer_params, policy_params, obs_batch)  # (grid_resolution^2, action_dim)
    
    # Compute Q-values for state-action-goal tuples
    q_values_pair = q_network.apply(normalizer_params, q_params, obs_batch, actions)  # (grid_resolution^2, 2)
    # Take minimum of twin Q-networks
    q_values = jnp.min(q_values_pair, axis=-1)  # (grid_resolution^2,)
    
    # Reshape back to grid
    q_grid = np.array(q_values.reshape(grid_resolution, grid_resolution))
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(q_grid, extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
                   origin='lower', cmap='viridis', aspect='equal')
    
    # Mark the current state position
    state_goal_pos = state[goal_indices]
    ax.plot(state_goal_pos[0], state_goal_pos[1], 'r*', markersize=20, 
            label=f'Current State: ({state_goal_pos[0]:.2f}, {state_goal_pos[1]:.2f})')
    
    ax.set_xlabel('Goal x')
    ax.set_ylabel('Goal y')
    ax.set_title(f'TD3 Q-Function Landscape\nState: [{state_goal_pos[0]:.2f}, {state_goal_pos[1]:.2f}]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Q-value', rotation=270, labelpad=20)
    
    # Save to buffer and log to WandB
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    pil_image = Image.open(buf)
    wandb.log({wandb_key: wandb.Image(pil_image)})


def visualize_sac_goals_2d(start_xy, proposed_goals_xy, final_states_xy, wandb_key,
                            intermediate_xy=None, x_bounds=None, y_bounds=None):
    '''Visualize 2D goals for SAC-style goal-conditioned RL.
    Shows trajectories from start state to final achieved state, with proposed goals.
    - start_xy: (num_samples, 2) array of start states
    - proposed_goals_xy: (num_samples, 2) array of proposed goals (from replay buffer)
    - final_states_xy: (num_samples, 2) array of final achieved states
    - wandb_key: str, key to log the plot in WandB
    - intermediate_xy: optional (num_samples, num_intermediate, 2) array of intermediate trajectory states
    - x_bounds: tuple (min, max) for x-axis range, or None for auto
    - y_bounds: tuple (min, max) for y-axis range, or None for auto
    '''
    assert start_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert proposed_goals_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert final_states_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    if intermediate_xy is not None:
        assert intermediate_xy.shape[2] == 2, "Intermediate trajectory visualization only supported for 2D"
    
    fig = go.Figure()
    
    num_samples = start_xy.shape[0]
    
    # Plot trajectories first (so points appear on top)
    for i in range(num_samples):
        # If we have intermediate states, plot full trajectory line through them
        if intermediate_xy is not None:
            # Build full trajectory: start -> intermediate -> final
            full_traj_xy = np.vstack([
                start_xy[i:i+1],
                intermediate_xy[i],
                final_states_xy[i:i+1]
            ])
            
            # Full trajectory line
            fig.add_trace(go.Scatter(
                x=full_traj_xy[:, 0],
                y=full_traj_xy[:, 1],
                mode='lines',
                line=dict(color='purple', width=1.5),
                opacity=0.4,
                showlegend=(i == 0),
                name='Trajectory' if i == 0 else '',
                hoverinfo='skip'
            ))
            
            # Intermediate trajectory points
            fig.add_trace(go.Scatter(
                x=intermediate_xy[i, :, 0],
                y=intermediate_xy[i, :, 1],
                mode='markers',
                marker=dict(color='purple', size=3, opacity=0.5),
                showlegend=(i == 0),
                name='Trajectory Points' if i == 0 else '',
                hovertemplate='Intermediate<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
            ))
        else:
            # Simple line from start state to final state
            fig.add_trace(go.Scatter(
                x=[start_xy[i, 0], final_states_xy[i, 0]],
                y=[start_xy[i, 1], final_states_xy[i, 1]],
                mode='lines',
                line=dict(color='purple', width=1.5),
                opacity=0.4,
                showlegend=(i == 0),
                name='Trajectory' if i == 0 else '',
                hoverinfo='skip'
            ))
        
        # Dashed line from proposed goal to final state (goal-achievement gap)
        fig.add_trace(go.Scatter(
            x=[proposed_goals_xy[i, 0], final_states_xy[i, 0]],
            y=[proposed_goals_xy[i, 1], final_states_xy[i, 1]],
            mode='lines',
            line=dict(color='orange', width=1, dash='dash'),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Plot start states (green circles)
    fig.add_trace(go.Scatter(
        x=start_xy[:, 0], y=start_xy[:, 1],
        mode='markers',
        marker=dict(color='green', size=10, symbol='circle'),
        name='Start State',
        hovertemplate='Start<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    # Plot proposed goals (blue diamonds)
    fig.add_trace(go.Scatter(
        x=proposed_goals_xy[:, 0], y=proposed_goals_xy[:, 1],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='diamond'),
        name='Proposed Goal',
        hovertemplate='Proposed Goal<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    # Plot final achieved states (red stars)
    fig.add_trace(go.Scatter(
        x=final_states_xy[:, 0], y=final_states_xy[:, 1],
        mode='markers',
        marker=dict(color='red', size=14, symbol='star'),
        name='Final State',
        hovertemplate='Final State<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='SAC Goal Proposals and Achieved Trajectories',
        xaxis_title='x',
        yaxis_title='y',
        xaxis=dict(range=x_bounds) if x_bounds else {},
        yaxis=dict(range=y_bounds) if y_bounds else {},
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='closest',
        width=800,
        height=800,
    )
    
    # Log to WandB as interactive plot
    wandb.log({wandb_key: fig})