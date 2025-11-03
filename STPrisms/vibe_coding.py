"""
Vibe Coding Layer

Provides interpretive overlay on analytical outputs,
capturing qualitative and epistemic aspects of spatial reasoning.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class VibeCategory(Enum):
    """Vibe tag categories"""
    # Epistemic stance
    CERTAINTY = "certainty"
    UNCERTAINTY = "uncertainty"
    POSSIBILITY = "possibility"
    IMPOSSIBILITY = "impossibility"
    AMBIGUITY = "ambiguity"
    
    # Constraint level
    FREE = "free"
    CONSTRAINED = "constrained"
    BLOCKED = "blocked"
    
    # Evidence tone
    STRONG_EVIDENCE = "strong"
    WEAK_EVIDENCE = "weak"
    CONFLICTING_EVIDENCE = "conflicting"
    
    # Perspective
    HUMAN_CENTERED = "human-centered"
    DATA_CENTERED = "data-centered"
    ALGORITHMIC = "algorithmic"
    
    # Narrative mood
    EXPLORATORY = "exploratory"
    FORMAL = "formal"
    APPLIED = "applied"
    SPECULATIVE = "speculative"


@dataclass
class VibeTag:
    """Single vibe tag with intensity"""
    category: VibeCategory
    intensity: float = 0.5  # 0.0 to 1.0
    description: str = ""
    
    def to_dict(self) -> dict:
        return {
            'category': self.category.value,
            'intensity': self.intensity,
            'description': self.description
        }


@dataclass
class VibeAnnotation:
    """
    Qualitative annotation on analytical result
    
    Links computational outputs with interpretive qualities
    """
    target_type: str  # "prism", "alibi_query", "probability_field", etc.
    target_id: str
    tags: List[VibeTag] = field(default_factory=list)
    author: str = "system"
    narrative: str = ""
    timestamp: Optional[str] = None
    
    def add_tag(self, category: VibeCategory, intensity: float = 0.5, description: str = ""):
        """Add a vibe tag"""
        self.tags.append(VibeTag(category, intensity, description))
    
    def get_dominant_vibe(self) -> Optional[VibeTag]:
        """Get tag with highest intensity"""
        if not self.tags:
            return None
        return max(self.tags, key=lambda t: t.intensity)
    
    def to_dict(self) -> dict:
        return {
            'target_type': self.target_type,
            'target_id': self.target_id,
            'tags': [t.to_dict() for t in self.tags],
            'author': self.author,
            'narrative': self.narrative,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VibeAnnotation':
        """Create from dictionary"""
        tags = []
        for t in data.get('tags', []):
            tags.append(VibeTag(
                VibeCategory(t['category']),
                t.get('intensity', 0.5),
                t.get('description', '')
            ))
        
        return cls(
            target_type=data['target_type'],
            target_id=data['target_id'],
            tags=tags,
            author=data.get('author', 'system'),
            narrative=data.get('narrative', ''),
            timestamp=data.get('timestamp')
        )


class VibeAnalyzer:
    """
    Automatic vibe tag generation from analytical results
    
    Maps quantitative metrics to qualitative interpretations
    """
    
    @staticmethod
    def analyze_prism(prism) -> VibeAnnotation:
        """
        Analyze space-time prism characteristics
        
        Returns vibe annotation based on prism geometry
        """
        from stprisms_core import SpaceTimePrism
        
        annotation = VibeAnnotation(
            target_type="prism",
            target_id=f"prism_{id(prism)}",
            author="auto-analyzer"
        )
        
        # Constraint level based on time budget utilization
        constraint_ratio = prism.spatial_distance() / (prism.vmax * prism.time_budget())
        
        if constraint_ratio > 0.9:
            annotation.add_tag(VibeCategory.BLOCKED, 0.9, "Tight time constraint")
            annotation.add_tag(VibeCategory.CERTAINTY, 0.8, "Trajectory highly determined")
        elif constraint_ratio > 0.6:
            annotation.add_tag(VibeCategory.CONSTRAINED, 0.7, "Moderate movement freedom")
            annotation.add_tag(VibeCategory.UNCERTAINTY, 0.4, "Some trajectory variability")
        else:
            annotation.add_tag(VibeCategory.FREE, 0.8, "High movement freedom")
            annotation.add_tag(VibeCategory.UNCERTAINTY, 0.7, "Many possible trajectories")
        
        # Feasibility
        if not prism.is_feasible():
            annotation.add_tag(VibeCategory.IMPOSSIBILITY, 1.0, "Physically impossible trajectory")
            annotation.add_tag(VibeCategory.CONFLICTING_EVIDENCE, 0.9, "Data inconsistency")
        
        return annotation
    
    @staticmethod
    def analyze_alibi_query(query_result: dict) -> VibeAnnotation:
        """
        Analyze alibi query results
        
        Interprets intersection probability and overlap characteristics
        """
        annotation = VibeAnnotation(
            target_type="alibi_query",
            target_id=f"query_{id(query_result)}",
            author="auto-analyzer"
        )
        
        prob = query_result['intersection_probability']
        exists = query_result['intersection_exists']
        area = query_result['intersection_area']
        
        # Epistemic stance
        if not exists:
            annotation.add_tag(VibeCategory.IMPOSSIBILITY, 1.0, "No temporal-spatial overlap")
            annotation.add_tag(VibeCategory.CERTAINTY, 0.9, "Objects could not meet")
            annotation.narrative = "Meeting impossible: no overlapping reachable regions."
        elif prob < 0.1:
            annotation.add_tag(VibeCategory.POSSIBILITY, 0.3, "Barely possible encounter")
            annotation.add_tag(VibeCategory.UNCERTAINTY, 0.9, "Very uncertain meeting")
            annotation.add_tag(VibeCategory.WEAK_EVIDENCE, 0.8, "Minimal overlap")
            annotation.narrative = f"Meeting barely possible: {prob*100:.1f}% temporal overlap."
        elif prob < 0.5:
            annotation.add_tag(VibeCategory.POSSIBILITY, 0.6, "Possible encounter")
            annotation.add_tag(VibeCategory.UNCERTAINTY, 0.6, "Uncertain meeting")
            annotation.add_tag(VibeCategory.AMBIGUITY, 0.5, "Ambiguous evidence")
            annotation.narrative = f"Meeting possible but uncertain: {prob*100:.1f}% overlap."
        else:
            annotation.add_tag(VibeCategory.POSSIBILITY, 0.9, "Likely encounter")
            annotation.add_tag(VibeCategory.STRONG_EVIDENCE, 0.7, "Substantial overlap")
            annotation.add_tag(VibeCategory.CERTAINTY, 0.6, "Probable meeting")
            annotation.narrative = f"Meeting likely: {prob*100:.1f}% temporal overlap."
        
        # Constraint based on area
        if area > 0:
            if area < 10:  # Small area in m²
                annotation.add_tag(VibeCategory.CONSTRAINED, 0.8, "Narrow meeting zone")
            elif area < 100:
                annotation.add_tag(VibeCategory.CONSTRAINED, 0.5, "Moderate meeting zone")
            else:
                annotation.add_tag(VibeCategory.FREE, 0.6, "Large meeting zone")
        
        return annotation
    
    @staticmethod
    def analyze_visit_probability(prob_field) -> VibeAnnotation:
        """
        Analyze visit probability field
        
        Interprets spatial accessibility patterns
        """
        annotation = VibeAnnotation(
            target_type="visit_probability",
            target_id=f"prob_field_{id(prob_field)}",
            author="auto-analyzer"
        )
        
        max_prob = prob_field['probability'].max()
        mean_prob = prob_field['probability'].mean()
        coverage = (prob_field['probability'] > 0.1).sum() / len(prob_field)
        
        # Accessibility
        if max_prob > 0.8:
            annotation.add_tag(VibeCategory.CERTAINTY, 0.7, "High certainty zone exists")
            annotation.add_tag(VibeCategory.STRONG_EVIDENCE, 0.6, "Clear visit pattern")
        else:
            annotation.add_tag(VibeCategory.UNCERTAINTY, 0.7, "Diffuse visit pattern")
            annotation.add_tag(VibeCategory.WEAK_EVIDENCE, 0.5, "Unclear accessibility")
        
        # Freedom
        if coverage > 0.5:
            annotation.add_tag(VibeCategory.FREE, 0.8, "Wide area accessible")
        elif coverage > 0.2:
            annotation.add_tag(VibeCategory.CONSTRAINED, 0.5, "Moderate accessibility")
        else:
            annotation.add_tag(VibeCategory.BLOCKED, 0.7, "Limited accessibility")
        
        annotation.narrative = f"Visit probability: max={max_prob:.2f}, mean={mean_prob:.2f}, coverage={coverage*100:.1f}%"
        
        return annotation


class VibeColorMapper:
    """
    Map vibe tags to visual encodings
    
    Provides color schemes and visual styles for different vibes
    """
    
    VIBE_COLORS = {
        # Epistemic stance - blues to reds
        VibeCategory.CERTAINTY: "#2E7D32",       # Green (certain)
        VibeCategory.UNCERTAINTY: "#F57C00",     # Orange (uncertain)
        VibeCategory.POSSIBILITY: "#1976D2",     # Blue (possible)
        VibeCategory.IMPOSSIBILITY: "#C62828",   # Red (impossible)
        VibeCategory.AMBIGUITY: "#7B1FA2",       # Purple (ambiguous)
        
        # Constraint level - grays to bright
        VibeCategory.FREE: "#4CAF50",            # Bright green (free)
        VibeCategory.CONSTRAINED: "#FF9800",     # Orange (constrained)
        VibeCategory.BLOCKED: "#424242",         # Dark gray (blocked)
        
        # Evidence tone - saturation levels
        VibeCategory.STRONG_EVIDENCE: "#1565C0",     # Deep blue
        VibeCategory.WEAK_EVIDENCE: "#90CAF9",       # Light blue
        VibeCategory.CONFLICTING_EVIDENCE: "#D32F2F", # Red
        
        # Perspective - cool to warm
        VibeCategory.HUMAN_CENTERED: "#E91E63",      # Pink
        VibeCategory.DATA_CENTERED: "#00ACC1",       # Cyan
        VibeCategory.ALGORITHMIC: "#5E35B1",         # Deep purple
        
        # Narrative mood - varied
        VibeCategory.EXPLORATORY: "#FBC02D",         # Yellow
        VibeCategory.FORMAL: "#455A64",              # Blue gray
        VibeCategory.APPLIED: "#43A047",             # Green
        VibeCategory.SPECULATIVE: "#AB47BC",         # Light purple
    }
    
    @classmethod
    def get_color(cls, vibe: VibeCategory) -> str:
        """Get color for vibe category"""
        return cls.VIBE_COLORS.get(vibe, "#999999")
    
    @classmethod
    def get_gradient_color(cls, vibe: VibeCategory, intensity: float) -> str:
        """
        Get color with opacity based on intensity
        
        Returns: rgba string
        """
        base_color = cls.get_color(vibe)
        # Convert hex to rgb
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        
        return f"rgba({r}, {g}, {b}, {intensity})"
    
    @classmethod
    def get_annotation_summary_color(cls, annotation: VibeAnnotation) -> str:
        """Get representative color for annotation (dominant vibe)"""
        dominant = annotation.get_dominant_vibe()
        if dominant is None:
            return "#999999"
        
        return cls.get_gradient_color(dominant.category, dominant.intensity)


class VibeNarrative:
    """
    Narrative workspace for interpretive annotations
    
    Manages researcher notes, hypotheses, and qualitative reflections
    """
    
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
    
    def add_entry(self, 
                  title: str, 
                  content: str, 
                  linked_annotations: List[str] = None,
                  author: str = "researcher"):
        """Add narrative entry"""
        entry = {
            'title': title,
            'content': content,
            'linked_annotations': linked_annotations or [],
            'author': author,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self.entries.append(entry)
    
    def get_entries(self) -> List[Dict[str, Any]]:
        """Get all narrative entries"""
        return self.entries
    
    def export_to_json(self, filepath: str):
        """Export narrative to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.entries, f, indent=2)
    
    def import_from_json(self, filepath: str):
        """Import narrative from JSON file"""
        with open(filepath, 'r') as f:
            self.entries = json.load(f)


# Import pandas for timestamp
import pandas as pd
