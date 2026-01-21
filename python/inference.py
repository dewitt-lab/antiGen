import sys
import csv
import torch
import torch.nn as nn
from main import FitnessPredictor  # Import your model class
from helpers import *

def load_trained_model(checkpoint_path, device='cuda'):
    """
    Load a trained fitness prediction model
    
    Args:
        checkpoint_path: Path to saved .pt file
        device: 'cuda' or 'cpu'
    
    Returns:
        model, parameters dict
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract hyperparameters
    d_model = checkpoint['d_model']
    N = checkpoint['N']
    E = checkpoint['E']
    seq_len = N // 3
    
    # Recreate model architecture
    model = FitnessPredictor(
        vocab_size=E,
        seq_len=seq_len,
        d_model=d_model,
        nhead=4,
        dim_feedforward=4*d_model,
        dropout=0.1
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Extract population genetics parameters
    params = {
        'lam': torch.exp(checkpoint['log_lam']).to(device),
        'gam': torch.exp(checkpoint['log_gam']).to(device),
        'psi': torch.exp(checkpoint['log_psi']).to(device),
        'mu': torch.exp(checkpoint['log_mu']).to(device),
        'f0': torch.exp(checkpoint['log_f0']).to(device),
    }
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Final loss: {checkpoint['loss']:.2f}")
    print(f"Parameters:")
    print(f"  mu: {params['mu'].item():.6f}")
    print(f"  psi: {params['psi'].item():.3f}")
    print(f"  f0: {params['f0'].item():.3f}")
    print(f"  R0: {(params['f0'] * params['lam'][-1] / params['gam']).item():.2f}")
    
    return model, params, checkpoint


def main():
    
    """
    Main runner function
    """

    # Define alphabet
    allowed_nucleotides = "ACGT-"
    D = len(allowed_nucleotides) # Alphabet size = embedding dimension for now. TO BE IMPROVED

    allowed_aas = "ACDEFGHIKLMNPQRSTVWY*-"
    E = len(allowed_aas)

    # Check that the file is being properly used
    if (len(sys.argv) !=3):
        print("Please specify a model file and sequence file.")
        return
    
    # Input files
    model_file = sys.argv[1]
    seq_file = sys.argv[2]

    with open(seq_file, 'r') as f:
        aa_seq = f.read()

    # Remove trailing newline:
    aa_seq = aa_seq.strip()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create torch tensor for this one sequence
    aa_tensor = create_one_hot_tensor(aa_seq, len(aa_seq), E, allowed_aas, device).unsqueeze(0)

    # Run inference
    fitness_model, params, checkpoint = load_trained_model(model_file, device)

    with torch.no_grad():
        # Apply transformer
        fitness = fitness_model(aa_tensor)
        fitness = torch.sigmoid(fitness)
        fitness = -torch.log(1.0 - fitness)

        # Zero out fitness for stop codon
        fitness[:, :, char_to_index('*', allowed_aas)] = 0.0

    # Remove batch dimension: (1, N/3, E) -> (N/3, E)
    fitness = fitness.squeeze(0)
    
    # Move to CPU and convert to numpy
    fitness_matrix = fitness.cpu().numpy()
    
    print(f"Fitness matrix shape: {fitness_matrix.shape}")
    
    # Save to CSV
    output_file = 'fitness_predictions.csv'
    
    print(f"Saving fitness predictions to {output_file}...")
    
    # Create header: Position, then all amino acids
    header = ['Position'] + list(allowed_aas)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        # Write each position (codon)
        for pos in range(len(aa_seq)):
            row = [pos + 1]  # Position (1-indexed)
            # Add fitness values for all possible amino acids
            row.extend(fitness_matrix[pos, :].tolist())
            writer.writerow(row)
    
    print(f"Successfully saved {len(aa_seq)} positions × {E} amino acids to {output_file}")
    
    


if __name__=="__main__":
    main()