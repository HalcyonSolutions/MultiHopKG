"""
Operations for embeddings in the knowledge graph, including angle operations.
"""
import torch

def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """
    Normalize an angle to the range [-π, π] using modulo arithmetic.

    This function ensures that input angles remain within a standard range
    by wrapping them using modulo operations.

    Args:
        angle (torch.Tensor): Angle in radians.

    Returns:
        torch.Tensor: Normalized angle in the range [-π, π].
    """
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


def normalize_angle_smooth(angle: torch.Tensor) -> torch.Tensor:
    """
    Smoothly normalize an angle to the range [-π, π] using trigonometric functions.

    This approach ensures differentiability across all values, making it
    suitable for gradient-based optimization tasks.

    Args:
        angle (torch.Tensor): Angle in radians.

    Returns:
        torch.Tensor: Smoothly normalized angle in the range [-π, π].
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def angular_difference(angle1: torch.Tensor, angle2: torch.Tensor, smooth: bool = True, use_abs: bool = False) -> torch.Tensor:
    """
    Compute the shortest angular difference between two angles in radians.

    The function calculates the absolute shortest distance between two angles
    while accounting for periodicity. Optionally, it can use a differentiable 
    normalization method.

    Args:
        angle1 (torch.Tensor): First angle in radians.
        angle2 (torch.Tensor): Second angle in radians.
        smooth (bool, optional): If True, uses a differentiable approach. Defaults to True.

    Returns:
        torch.Tensor: The absolute shortest distance between the two angles in radians.
    """
    diff = normalize_angle_smooth(angle2 - angle1) if smooth else normalize_angle(angle2 - angle1)
    if use_abs:
        return torch.abs(diff)
    return diff


def total_angular_displacement(angle1: torch.Tensor, angle2: torch.Tensor) -> torch.Tensor:
    """
    Compute the total angular displacement (sum of shortest angular differences).

    This function calculates the sum of angular distances across the last dimension,
    which can be useful for measuring cumulative rotational differences.

    Args:
        angle1 (torch.Tensor): Tensor of first angles in radians.
        angle2 (torch.Tensor): Tensor of second angles in radians.

    Returns:
        torch.Tensor: Sum of angular differences along the last dimension.
    """
    return angular_difference(angle1, angle2, use_abs = True).sum(dim=-1)

def cosine_similarity(A: torch.Tensor, B: torch.Tensor):
    """
    Compute cosine similarity between two complex-valued vectors.
    A, B: (vector_dim,) - tensor where first half of the last dimension is the real part and the second half is the imaginary part
    """
    A = torch.complex(*torch.chunk(A, 2))
    B = torch.complex(*torch.chunk(B, 2))

    # Compute complex modulus (L2 norm) for each vector
    norm_A = torch.norm(A, p=2)
    norm_B = torch.norm(B, p=2)

    # Compute Hermitian inner product (dot product with conjugate)
    inner_product = torch.sum(A * B.conj())

    # Extract real part of similarity
    return torch.real(inner_product) / (norm_A * norm_B + 1e-9)  # Avoid division by zero

def chamfer_distance_consine(A: torch.Tensor, B: torch.Tensor):
    """
    Compute Chamfer Distance between two sets of complex-valued vectors using Complex Cosine Similarity.
    A: (batch, num_vectors_A, vector_dim) - tensor where first half of the last dimension is the real part and the second half is the imaginary part
    B: (batch, num_vectors_B, vector_dim) - tensor where first half of the last dimension is the real part and the second half is the imaginary part
    """
    distance_matrix = _chamfer_distance_cosine_part1(A, B)

    return _chamfer_distance_cosine_part2(distance_matrix)

def _chamfer_distance_cosine_part1(A, B):
    # Convert real-imaginary concatenated format to complex tensor
    A = torch.complex(*torch.chunk(A, 2, dim=-1)) # Expected shape: (batch, num_vectors_A, vector_dim//2)
    B = torch.complex(*torch.chunk(B, 2, dim=-1))  # Expected shape: (batch, num_vectors_B, vector_dim//2)

    # Normalize using complex modulus (L2 norm for complex numbers)
    A_norm = A / (torch.norm(A, p=2, dim=-1, keepdim=True) + 1e-9)  # Expected shape: (batch, num_vectors_A, vector_dim//2)
    B_norm = B / (torch.norm(B, p=2, dim=-1, keepdim=True) + 1e-9)  # Expected shape: (batch, num_vectors_B, vector_dim//2)

    # Compute Hermitian inner product (cosine similarity in complex space)
    similarity_matrix = torch.real(torch.matmul(A_norm, B_norm.conj().transpose(1, 2)))  # Re(A * B^H)
    # Expected shape: (batch, num_vectors_A, num_vectors_B) - Pairwise cosine similarities

    distance_matrix = 1 - similarity_matrix  # Convert similarity to distance
    return distance_matrix

def _chamfer_distance_cosine_part2(distance_matrix):
    # Chamfer Distance: Find closest match for each point
    min_A_to_B, _ = torch.min(distance_matrix, dim=2)  # Expected shape: (batch, num_vectors_A) - Min distance for each A
    min_B_to_A, _ = torch.min(distance_matrix, dim=1)  # Expected shape: (batch, num_vectors_B) - Min distance for each B
    return min_A_to_B.mean(dim=-1) + min_B_to_A.mean(dim=-1)  # Symmetric Chamfer Distance
    # return min_A_to_B.mean() + min_B_to_A.mean()  # Scalar loss value (single float), Symmetric Chamfer Distance

def chamfer_distance(A, B):
    """
    Compute the Chamfer Distance between two sets of vectors.
    A: (batch, num_vectors_A, vector_dim), in our case (batch, step, embedding_dim) for the visited embeddings
    B: (batch, num_vectors_B, vector_dim), in our case (batch, num_relevant_entities, embedding_dim) for the relevant embeddings
    """
    
    distances = _chamfer_distance_part1(A, B)
    
    # NOTE: torch.min is differiantiable ONLY for the value, NOT the index

    return _chamfer_distance_part2(distances)

def _chamfer_distance_part1(A, B):
    A = A.unsqueeze(2)  # Shape: (batch, num_vectors_A, 1, vector_dim)
    B = B.unsqueeze(1)  # Shape: (batch, 1, num_vectors_B, vector_dim)

    # ! TODO: Correct this so that it has a more compatible distance metric for RotatE
    return torch.norm(A - B, dim=-1)  # Compute pairwise Euclidean distances

def _chamfer_distance_part2(distances):
    min_A_to_B, _ = torch.min(distances, dim=2)  # For each A, find nearest B
    min_B_to_A, _ = torch.min(distances, dim=1)  # For each B, find nearest A
    
    return min_A_to_B.mean(dim=(1,2)) + min_B_to_A.mean(dim=(1,2))  # Symmetric Chamfer Distance
    # return min_A_to_B.mean() + min_B_to_A.mean()  # Symmetric Chamfer Distance