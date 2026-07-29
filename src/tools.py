"""
Structured Pokédex & Learnset Tool module for VGC RAG.
Queries live PokeAPI endpoints for hard factual data (Base Stats, Typing, Learnsets)
to eliminate hallucinations on structured game mechanics.
"""
import requests
from typing import Optional

def clean_name(name: str) -> str:
    """Formats Pokemon or Move names for PokeAPI URLs (e.g., 'Flutter Mane' -> 'flutter-mane')."""
    return name.lower().strip().replace(" ", "-").replace(".", "").replace("'", "")

def lookup_pokedex_fact(pokemon_name: str, move_to_check: Optional[str] = None) -> str:
    """
    Queries PokeAPI for a Pokemon's exact typing, base stats, abilities, 
    and verifies if they can learn a specific move.
    
    Args:
        pokemon_name (str): Name of the Pokemon (e.g., 'incineroar', 'flutter-mane')
        move_to_check (str, optional): Name of a specific move to verify (e.g., 'parting-shot')
        
    Returns:
        str: A formatted, authoritative factual summary block.
    """
    clean_pkmn = clean_name(pokemon_name)
    url = f"https://pokeapi.co/api/v2/pokemon/{clean_pkmn}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return f"[TOOL ERROR] Could not find structured Pokédex data for '{pokemon_name}' in PokeAPI."
            
        data = response.json()
        
        # 1. Extract Types
        types = [t["type"]["name"].capitalize() for t in data.get("types", [])]
        type_str = "/".join(types)
        
        # 2. Extract Abilities
        abilities = [
            f"{a['ability']['name'].capitalize()}{' (Hidden)' if a['is_hidden'] else ''}" 
            for a in data.get("abilities", [])
        ]
        ability_str = ", ".join(abilities)
        
        # 3. Extract Base Stats
        stats = {s["stat"]["name"]: s["base_stat"] for s in data.get("stats", [])}
        stat_str = (
            f"HP: {stats.get('hp', '?')} | Atk: {stats.get('attack', '?')} | "
            f"Def: {stats.get('defense', '?')} | SpA: {stats.get('special-attack', '?')} | "
            f"SpD: {stats.get('special-defense', '?')} | Spe: {stats.get('speed', '?')}"
        )
        
        result_block = (
            f"[AUTHORITATIVE POKEAPI FACT] {pokemon_name.capitalize()}:\n"
            f"- Typing: {type_str}\n"
            f"- Abilities: {ability_str}\n"
            f"- Base Stats: {stat_str}\n"
        )
        
        # 4. Check Learnset if a specific move was requested
        if move_to_check:
            clean_mv = clean_name(move_to_check)
            all_moves = [m["move"]["name"] for m in data.get("moves", [])]
            
            if clean_mv in all_moves:
                result_block += f"- Learnset Verification: YES, {pokemon_name.capitalize()} legitimately learns the move '{move_to_check.upper()}'."
            else:
                result_block += f"- Learnset Verification: NO, {pokemon_name.capitalize()} CANNOT learn the move '{move_to_check.upper()}'. Do not suggest it."
                
        return result_block
        
    except Exception as e:
        return f"[TOOL ERROR] Network failure querying PokeAPI: {str(e)}"

# Quick debugging block
if __name__ == "__main__":
    print(lookup_pokedex_fact("flutter-mane", move_to_check="trick-room"))
    print("\n" + "="*50 + "\n")
    print(lookup_pokedex_fact("incineroar", move_to_check="parting-shot"))