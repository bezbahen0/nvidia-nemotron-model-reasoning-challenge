import pandas as pd


def upsample_dataframe(df, multipliers_map, label_col="label", seed=42):
    upsampled_pieces = []
    
    for task_name, group in df.groupby(label_col):
        multiplier = multipliers_map.get(task_name, 1.0)
        
        full_repeats = int(multiplier)
        fraction = multiplier - full_repeats
        
        if full_repeats > 0:
            upsampled_pieces.append(pd.concat([group] * full_repeats))
            
        if fraction > 0:
            fractional_group = group.sample(frac=fraction, random_state=seed)
            upsampled_pieces.append(fractional_group)
            
    final_df = pd.concat(upsampled_pieces, ignore_index=True)
    
    final_df = final_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    return final_df
