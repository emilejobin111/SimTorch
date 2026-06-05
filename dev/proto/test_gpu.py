import jax
import jax.numpy as jnp
import time

def run_test():
    print("=== 🔍 Vérification du Matériel ===")
    
    # 1. Lister les périphériques disponibles
    devices = jax.devices()
    print(f"Périphériques détectés : {devices}")
    
    # Vérification stricte du GPU
    if devices[0].platform == "gpu":
        print("✅ SUCCÈS : JAX a trouvé votre carte graphique NVIDIA !")
    else:
        print("❌ ERREUR : JAX tourne sur le CPU. Quelque chose s'est mal passé.")
        return

    print("\n=== 🚀 Test de Calcul (Multiplication Matricielle) ===")
    
    # 2. Générer deux grandes matrices (5000 x 5000)
    taille = 100
    print(f"Génération de deux matrices aléatoires de {taille}x{taille} donc de {taille**2} nombre par matrice")
    key = jax.random.PRNGKey(42)
    print(key)
    matrice_A = jax.random.normal(key, (taille, taille),dtype=jnp.bfloat16)
    matrice_B = jax.random.normal(key, (taille, taille),dtype=jnp.bfloat16)

    # 3. Définir une fonction et la compiler avec @jax.jit pour des performances maximales
    @jax.jit
    def calcul_lourd(a, b)->jnp.ndarray:
        return a @ b

    # 4. Phase de chauffe (Warmup)
    # JAX compile la fonction lors de sa première exécution. On le fait une fois dans le vide.
    print("Compilation JIT en cours (warmup)...")
    _ = calcul_lourd(matrice_A, matrice_B).block_until_ready()

    # 5. Le vrai test de vitesse
    print("Calcul sur la RTX 4060 en cours...")
    debut = time.time()
    
    # L'opération matricielle réelle
    resultat : jnp.ndarray = calcul_lourd(matrice_A, matrice_B)
    
    # JAX est asynchrone par défaut ! Il faut lui dire d'attendre la fin du calcul.
    resultat.block_until_ready() 
    
    fin = time.time()

    print(f"\n⚡ Opération terminée avec succès en {fin - debut:.4f} secondes !")
    print("Votre environnement est 100% prêt pour le Machine Learning.")

if __name__ == "__main__":
    run_test()