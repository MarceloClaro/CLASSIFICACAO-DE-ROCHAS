"""
3D Visualization Module
Provides interactive 3D visualizations for feature analysis and PCA
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from typing import List, Optional
import streamlit as st


def visualize_pca_3d(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title: str = "Análise PCA 3D das Features"
) -> go.Figure:
    """
    Create interactive 3D PCA visualization using Plotly
    
    Args:
        features: Feature array (n_samples, n_features)
        labels: Label array (n_samples,)
        class_names: List of class names
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    # Apply PCA
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features)
    
    # Create DataFrame-like structure for plotting
    data_dict = {
        'PC1': features_pca[:, 0],
        'PC2': features_pca[:, 1],
        'PC3': features_pca[:, 2],
        'Classe': [class_names[label] for label in labels],
        'Label': labels
    }
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Add trace for each class
    for i, class_name in enumerate(class_names):
        mask = labels == i
        fig.add_trace(go.Scatter3d(
            x=features_pca[mask, 0],
            y=features_pca[mask, 1],
            z=features_pca[mask, 2],
            mode='markers',
            name=class_name,
            marker=dict(
                size=6,
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            text=[f"Classe: {class_name}<br>PC1: {features_pca[j, 0]:.2f}<br>PC2: {features_pca[j, 1]:.2f}<br>PC3: {features_pca[j, 2]:.2f}" 
                  for j in np.where(mask)[0]],
            hoverinfo='text'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=900,
        height=700,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    # Add variance explained text
    total_var = pca.explained_variance_ratio_.sum() * 100
    fig.add_annotation(
        text=f"Variância Total Explicada: {total_var:.2f}%",
        xref="paper", yref="paper",
        x=0.5, y=0.02,
        showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
    
    return fig


def visualize_activation_heatmap_3d(
    activation_map: np.ndarray,
    title: str = "Mapa de Ativação 3D"
) -> go.Figure:
    """
    Create 3D surface plot of activation heatmap
    
    Args:
        activation_map: 2D activation map (height, width)
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    # Normalize activation map
    if activation_map.max() > 0:
        activation_norm = activation_map / activation_map.max()
    else:
        activation_norm = activation_map
    
    # Create meshgrid
    height, width = activation_norm.shape
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=activation_norm,
        colorscale='Hot',
        reversescale=True,
        colorbar=dict(
            title="Ativação",
            titleside="right"
        ),
        hovertemplate='X: %{x}<br>Y: %{y}<br>Ativação: %{z:.3f}<extra></extra>'
    )])
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Largura',
            yaxis_title='Altura',
            zaxis_title='Intensidade de Ativação',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=height/width, z=0.5)
        ),
        width=800,
        height=700
    )
    
    return fig


def visualize_feature_importance_3d(
    feature_importances: dict,
    title: str = "Importância das Features em 3D"
) -> go.Figure:
    """
    Create 3D bar chart for feature importance
    
    Args:
        feature_importances: Dictionary of feature names and importance values
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    features = list(feature_importances.keys())
    importances = list(feature_importances.values())
    
    # Create positions for 3D bars
    n_features = len(features)
    x_pos = np.arange(n_features)
    y_pos = np.zeros(n_features)
    z_pos = np.zeros(n_features)
    
    # Bar dimensions
    dx = np.ones(n_features) * 0.8
    dy = np.ones(n_features) * 0.8
    dz = np.array(importances)
    
    # Create 3D bar chart
    fig = go.Figure(data=[go.Mesh3d(
        x=[],
        y=[],
        z=[],
        opacity=0.8,
        color='lightblue'
    )])
    
    # Add bars manually
    colors = px.colors.sequential.Viridis
    for i in range(n_features):
        fig.add_trace(go.Mesh3d(
            x=[x_pos[i], x_pos[i], x_pos[i]+dx[i], x_pos[i]+dx[i], 
               x_pos[i], x_pos[i], x_pos[i]+dx[i], x_pos[i]+dx[i]],
            y=[y_pos[i], y_pos[i]+dy[i], y_pos[i]+dy[i], y_pos[i],
               y_pos[i], y_pos[i]+dy[i], y_pos[i]+dy[i], y_pos[i]],
            z=[z_pos[i], z_pos[i], z_pos[i], z_pos[i],
               z_pos[i]+dz[i], z_pos[i]+dz[i], z_pos[i]+dz[i], z_pos[i]+dz[i]],
            i=[0, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[1, 2, 3, 4, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[2, 3, 4, 5, 6, 7, 1, 6, 5, 5, 7, 6],
            name=features[i],
            color=colors[i % len(colors)],
            opacity=0.8,
            hovertemplate=f'{features[i]}<br>Importância: {importances[i]:.3f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                ticktext=features,
                tickvals=x_pos + dx/2,
                title="Features"
            ),
            yaxis_title='',
            zaxis_title='Importância',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=900,
        height=700,
        showlegend=True
    )
    
    return fig


def visualize_confusion_matrix_3d(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    title: str = "Matriz de Confusão 3D"
) -> go.Figure:
    """
    Create 3D visualization of confusion matrix
    
    Args:
        confusion_matrix: Confusion matrix (n_classes, n_classes)
        class_names: List of class names
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    n_classes = len(class_names)
    
    # Create meshgrid for positions
    x_pos, y_pos = np.meshgrid(np.arange(n_classes), np.arange(n_classes))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros(n_classes * n_classes)
    
    # Bar heights
    dz = confusion_matrix.flatten()
    
    # Colors based on diagonal (correct predictions vs errors)
    colors = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                colors.append('green')
            else:
                colors.append('red')
    
    # Create 3D bar chart
    fig = go.Figure()
    
    dx = 0.8
    dy = 0.8
    for idx in range(len(x_pos)):
        i, j = idx // n_classes, idx % n_classes
        fig.add_trace(go.Mesh3d(
            x=[x_pos[idx], x_pos[idx], x_pos[idx]+dx, x_pos[idx]+dx,
               x_pos[idx], x_pos[idx], x_pos[idx]+dx, x_pos[idx]+dx],
            y=[y_pos[idx], y_pos[idx]+dy, y_pos[idx]+dy, y_pos[idx],
               y_pos[idx], y_pos[idx]+dy, y_pos[idx]+dy, y_pos[idx]],
            z=[z_pos[idx], z_pos[idx], z_pos[idx], z_pos[idx],
               z_pos[idx]+dz[idx], z_pos[idx]+dz[idx], z_pos[idx]+dz[idx], z_pos[idx]+dz[idx]],
            i=[0, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[1, 2, 3, 4, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[2, 3, 4, 5, 6, 7, 1, 6, 5, 5, 7, 6],
            name=f'{class_names[i]} → {class_names[j]}',
            color=colors[idx],
            opacity=0.7,
            hovertemplate=f'Real: {class_names[i]}<br>Predito: {class_names[j]}<br>Count: {int(dz[idx])}<extra></extra>',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                ticktext=class_names,
                tickvals=np.arange(n_classes) + 0.4,
                title="Classe Predita"
            ),
            yaxis=dict(
                ticktext=class_names,
                tickvals=np.arange(n_classes) + 0.4,
                title="Classe Real"
            ),
            zaxis_title='Contagem',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=900,
        height=700
    )
    
    return fig


def create_interactive_3d_visualization(
    data_type: str,
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    **kwargs
) -> go.Figure:
    """
    Unified function to create various 3D visualizations
    
    Args:
        data_type: Type of visualization ('pca', 'heatmap', 'confusion', 'feature_importance')
        data: Data to visualize
        labels: Optional labels for data points
        class_names: Optional class names
        **kwargs: Additional arguments for specific visualizations
    
    Returns:
        Plotly figure object
    """
    if data_type == 'pca' and labels is not None and class_names is not None:
        return visualize_pca_3d(data, labels, class_names, **kwargs)
    elif data_type == 'heatmap':
        return visualize_activation_heatmap_3d(data, **kwargs)
    elif data_type == 'confusion' and class_names is not None:
        return visualize_confusion_matrix_3d(data, class_names, **kwargs)
    elif data_type == 'feature_importance':
        # data should be a dictionary
        return visualize_feature_importance_3d(data, **kwargs)
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")
