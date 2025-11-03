"""
Space-Time Prisms Core Models

Implements uncertainty-based trajectory analysis constructs from
Arthur Jansen's PhD thesis.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import pandas as pd


@dataclass
class Anchor:
    """Spatial-temporal anchor point with optional uncertainty"""
    x: float
    y: float
    t: float
    error_radius: float = 0.0  # Spatial uncertainty
    
    def as_point(self) -> Point:
        return Point(self.x, self.y)
    
    def distance_to(self, other: 'Anchor') -> float:
        """Euclidean distance to another anchor"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class TrajectorySample:
    """Sequence of observed anchor points"""
    object_id: str
    anchors: List[Anchor]
    vmax: float = 1.0  # Maximum velocity (m/s)
    color: str = "#1f77b4"
    
    def __post_init__(self):
        # Sort anchors by time
        self.anchors.sort(key=lambda a: a.t)
    
    def duration(self) -> float:
        """Total time span"""
        if len(self.anchors) < 2:
            return 0.0
        return self.anchors[-1].t - self.anchors[0].t
    
    def length(self) -> float:
        """Approximation of path length"""
        if len(self.anchors) < 2:
            return 0.0
        return sum(
            self.anchors[i].distance_to(self.anchors[i+1])
            for i in range(len(self.anchors) - 1)
        )


@dataclass
class SpaceTimePrism:
    """
    Space-time prism: reachable space between two anchors
    
    A prism is the intersection of two cones:
    - Forward cone from (p-, t-)
    - Backward cone to (p+, t+)
    """
    anchor_start: Anchor  # p-, t-
    anchor_end: Anchor    # p+, t+
    vmax: float           # Maximum velocity
    
    def time_budget(self) -> float:
        """Available time between anchors"""
        return self.anchor_end.t - self.anchor_start.t
    
    def spatial_distance(self) -> float:
        """Distance between anchor locations"""
        return self.anchor_start.distance_to(self.anchor_end)
    
    def is_feasible(self) -> bool:
        """Check if prism is geometrically valid"""
        return self.spatial_distance() <= self.vmax * self.time_budget()
    
    def ppa_at_time(self, t: float) -> Optional[Polygon]:
        """
        Potential Path Area (PPA) at time t
        
        Returns circular region of possible locations at time t,
        or None if t is outside prism bounds.
        """
        if t < self.anchor_start.t or t > self.anchor_end.t:
            return None
        
        # Time from start and end
        dt_start = t - self.anchor_start.t
        dt_end = self.anchor_end.t - t
        
        # Maximum distances from each anchor
        r_start = self.vmax * dt_start
        r_end = self.vmax * dt_end
        
        # Create circles around anchors
        circle_start = self.anchor_start.as_point().buffer(r_start)
        circle_end = self.anchor_end.as_point().buffer(r_end)
        
        # Intersection is the PPA
        ppa = circle_start.intersection(circle_end)
        
        if ppa.is_empty:
            return None
        
        return ppa
    
    def get_ppa_radius_at_time(self, t: float) -> Tuple[float, float]:
        """
        Get radii of forward and backward reachable sets at time t
        Returns: (r_start, r_end)
        """
        if t < self.anchor_start.t or t > self.anchor_end.t:
            return (0.0, 0.0)
        
        dt_start = t - self.anchor_start.t
        dt_end = self.anchor_end.t - t
        
        return (self.vmax * dt_start, self.vmax * dt_end)
    
    def compute_3d_cone_points(self, num_circle_points: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D mesh points for visualization
        Returns: (vertices, faces) for triangular mesh
        """
        # Generate circle points
        theta = np.linspace(0, 2*np.pi, num_circle_points, endpoint=False)
        
        vertices = []
        faces = []
        
        # Start apex
        vertices.append([self.anchor_start.x, self.anchor_start.y, self.anchor_start.t])
        
        # End apex
        vertices.append([self.anchor_end.x, self.anchor_end.y, self.anchor_end.t])
        
        # Sample times between anchors
        num_time_slices = 20
        times = np.linspace(self.anchor_start.t, self.anchor_end.t, num_time_slices)
        
        for t in times[1:-1]:  # Skip start and end
            ppa = self.ppa_at_time(t)
            if ppa is None or ppa.is_empty:
                continue
            
            # Get centroid and approximate radius
            centroid = ppa.centroid
            radius = np.sqrt(ppa.area / np.pi)
            
            # Create circle at this time slice
            for th in theta:
                x = centroid.x + radius * np.cos(th)
                y = centroid.y + radius * np.sin(th)
                vertices.append([x, y, t])
        
        return np.array(vertices), []


@dataclass
class PrismChain:
    """Chain of connected space-time prisms"""
    prisms: List[SpaceTimePrism]
    trajectory: TrajectorySample
    
    def total_freedom(self) -> float:
        """
        Aggregate measure of movement freedom
        Sum of PPA areas at midpoint times
        """
        freedom = 0.0
        for prism in self.prisms:
            t_mid = (prism.anchor_start.t + prism.anchor_end.t) / 2
            ppa = prism.ppa_at_time(t_mid)
            if ppa is not None:
                freedom += ppa.area
        return freedom
    
    def constraint_ratio(self) -> float:
        """
        Ratio of required travel to maximum possible
        Higher = more constrained
        """
        if not self.prisms:
            return 0.0
        
        total_required = sum(p.spatial_distance() for p in self.prisms)
        total_budget = sum(p.time_budget() * p.vmax for p in self.prisms)
        
        if total_budget == 0:
            return 1.0
        
        return total_required / total_budget


class AlibiQuery:
    """
    Alibi query: determine if multiple objects could have met
    
    Computes intersection of space-time prisms across objects
    """
    
    def __init__(self, trajectories: List[TrajectorySample]):
        self.trajectories = trajectories
        self.prism_chains = self._build_chains()
    
    def _build_chains(self) -> List[PrismChain]:
        """Build prism chains for each trajectory"""
        chains = []
        for traj in self.trajectories:
            prisms = []
            for i in range(len(traj.anchors) - 1):
                prism = SpaceTimePrism(
                    traj.anchors[i],
                    traj.anchors[i+1],
                    traj.vmax
                )
                if prism.is_feasible():
                    prisms.append(prism)
            chains.append(PrismChain(prisms, traj))
        return chains
    
    def evaluate(self, t_start: float, t_end: float, time_resolution: int = 50) -> dict:
        """
        Evaluate alibi query over time interval
        
        Returns:
            - intersection_exists: bool
            - intersection_area: float (maximum area)
            - intersection_probability: float (proportion of time slices with overlap)
            - time_slices: list of (time, area, polygon)
        """
        times = np.linspace(t_start, t_end, time_resolution)
        
        intersection_count = 0
        max_area = 0.0
        time_slices = []
        
        for t in times:
            # Get PPAs for all objects at time t
            ppas = []
            for chain in self.prism_chains:
                for prism in chain.prisms:
                    ppa = prism.ppa_at_time(t)
                    if ppa is not None and not ppa.is_empty:
                        ppas.append(ppa)
                        break  # Only need one PPA per object at this time
            
            if len(ppas) < len(self.trajectories):
                # Not all objects have valid PPAs at this time
                time_slices.append((t, 0.0, None))
                continue
            
            # Compute intersection
            intersection = ppas[0]
            for ppa in ppas[1:]:
                intersection = intersection.intersection(ppa)
            
            if not intersection.is_empty:
                area = intersection.area
                intersection_count += 1
                max_area = max(max_area, area)
                time_slices.append((t, area, intersection))
            else:
                time_slices.append((t, 0.0, None))
        
        return {
            'intersection_exists': intersection_count > 0,
            'intersection_area': max_area,
            'intersection_probability': intersection_count / len(times),
            'time_slices': time_slices,
            'num_objects': len(self.trajectories)
        }


class VisitProbability:
    """
    Compute visit probability for a location given a space-time prism
    
    Assumes uniform probability over all valid trajectories
    """
    
    def __init__(self, prism: SpaceTimePrism, grid_resolution: int = 50):
        self.prism = prism
        self.grid_resolution = grid_resolution
        
    def compute_probability_field(self) -> pd.DataFrame:
        """
        Compute probability density over spatial grid
        
        Returns DataFrame with columns: x, y, probability
        """
        # Determine spatial bounds
        x_min = min(self.prism.anchor_start.x, self.prism.anchor_end.x) - self.prism.vmax * self.prism.time_budget()
        x_max = max(self.prism.anchor_start.x, self.prism.anchor_end.x) + self.prism.vmax * self.prism.time_budget()
        y_min = min(self.prism.anchor_start.y, self.prism.anchor_end.y) - self.prism.vmax * self.prism.time_budget()
        y_max = max(self.prism.anchor_start.y, self.prism.anchor_end.y) + self.prism.vmax * self.prism.time_budget()
        
        # Create grid
        x_grid = np.linspace(x_min, x_max, self.grid_resolution)
        y_grid = np.linspace(y_min, y_max, self.grid_resolution)
        
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        
        # Sample times within prism
        num_time_samples = 20
        times = np.linspace(self.prism.anchor_start.t, self.prism.anchor_end.t, num_time_samples)
        
        # For each grid point, check if it's within PPA at any time
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                point = Point(x, y)
                visit_count = 0
                
                for t in times:
                    ppa = self.prism.ppa_at_time(t)
                    if ppa is not None and ppa.contains(point):
                        visit_count += 1
                
                Z[j, i] = visit_count / len(times)
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'x': X.flatten(),
            'y': Y.flatten(),
            'probability': Z.flatten()
        })
        
        return df


def create_trajectory_from_df(df: pd.DataFrame, object_id: str, vmax: float = 1.0) -> TrajectorySample:
    """
    Create TrajectorySample from DataFrame
    
    Expected columns: timestamp, x, y, [error_radius]
    """
    anchors = []
    for _, row in df.iterrows():
        error = row.get('error_radius', 0.0)
        anchor = Anchor(
            x=row['x'],
            y=row['y'],
            t=row['timestamp'],
            error_radius=error
        )
        anchors.append(anchor)
    
    return TrajectorySample(
        object_id=object_id,
        anchors=anchors,
        vmax=vmax
    )
