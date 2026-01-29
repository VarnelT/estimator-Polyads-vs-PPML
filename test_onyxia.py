import sys
print("Python executable:", sys.executable)

try:
    import polyads
    print("✅ VICTOIRE : Polyads est installé et prêt sur Onyxia !")
except ImportError as e:
    print("❌ ERREUR :", e)