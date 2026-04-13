"""
Blackjack Lab - Simülasyon & Veri Seti Üretici
100.000 el oynayıp her eli CSV'ye kaydeder.
Farklı stratejilerle simülasyon yapabilir.
"""

import csv
import os
import time
from game_engine import Deck, BlackjackGame


# ---- Stratejiler ----

def random_strategy(state):
    """Tamamen rastgele oyna."""
    import random
    return random.choice(["hit", "stand"])


def naive_strategy(state):
    """Basit kural: 17'nin altında hit, üstünde stand."""
    if state["player_total"] < 17:
        return "hit"
    return "stand"


def basic_strategy(state):
    """
    Kitaplardaki Basic Strategy tablosunun basitleştirilmiş hali.
    Gerçek casino stratejisine yakın kararlar verir.
    """
    player = state["player_total"]
    dealer = state["dealer_showing"]
    ace = state["usable_ace"]

    # Soft hand (kullanılabilir As var)
    if ace:
        if player <= 17:
            return "hit"
        elif player == 18:
            # Kasanın kartı 9, 10 veya As ise hit
            if dealer >= 9:
                return "hit"
            return "stand"
        else:
            return "stand"

    # Hard hand (As yok)
    if player <= 11:
        return "hit"
    elif player == 12:
        # Kasanın kartı 4-6 arası ise stand, değilse hit
        if 4 <= dealer <= 6:
            return "stand"
        return "hit"
    elif 13 <= player <= 16:
        # Kasanın kartı 2-6 arası ise stand, değilse hit
        if dealer <= 6:
            return "stand"
        return "hit"
    else:  # 17+
        return "stand"


# ---- Simülasyon ----

def simulate(num_games=100000, strategy_fn=None, strategy_name="random", num_decks=6):
    """
    Belirtilen sayıda el oynar ve sonuçları döndürür.

    Args:
        num_games: Oynanacak el sayısı
        strategy_fn: Strateji fonksiyonu
        strategy_name: CSV'de görünecek strateji adı
        num_decks: Deste sayısı

    Returns:
        list: Her elin bilgilerini içeren sözlük listesi
    """
    if strategy_fn is None:
        strategy_fn = random_strategy

    deck = Deck(num_decks=num_decks)
    results = []

    print(f"\n🎰 Simülasyon başlıyor: {num_games:,} el | Strateji: {strategy_name}")
    start_time = time.time()

    for i in range(num_games):
        game = BlackjackGame(deck)
        result = game.play_one_hand(strategy_fn)
        result["episode_id"] = i + 1
        result["strategy"] = strategy_name
        results.append(result)

        # İlerleme göstergesi
        if (i + 1) % 25000 == 0:
            elapsed = time.time() - start_time
            print(f"   ✓ {i+1:>7,} / {num_games:,} el tamamlandı ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    print(f"   ✅ Tamamlandı! ({elapsed:.1f}s)")

    return results


def save_to_csv(results, filepath="dataset/games.csv"):
    """Sonuçları CSV dosyasına kaydet."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fieldnames = [
        "episode_id", "strategy", "player_cards", "dealer_cards",
        "player_total", "dealer_total", "dealer_showing",
        "usable_ace", "num_hits", "actions", "result", "reward"
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📁 Veri seti kaydedildi: {filepath}")
    print(f"   Toplam satır: {len(results):,}")
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    print(f"   Dosya boyutu: {file_size:.1f} MB")


def print_stats(results, strategy_name):
    """Simülasyon istatistiklerini yazdır."""
    total = len(results)
    wins = sum(1 for r in results if r["result"] == "win")
    blackjacks = sum(1 for r in results if r["result"] == "blackjack")
    losses = sum(1 for r in results if r["result"] == "lose")
    draws = sum(1 for r in results if r["result"] == "draw")

    total_reward = sum(r["reward"] for r in results)
    avg_reward = total_reward / total

    print(f"\n📊 {strategy_name} Strateji İstatistikleri:")
    print(f"   Galibiyet:   {wins + blackjacks:>6,} ({(wins + blackjacks) / total * 100:.1f}%)")
    print(f"     ├─ Normal:  {wins:>6,}")
    print(f"     └─ BJ:      {blackjacks:>6,}")
    print(f"   Mağlubiyet:  {losses:>6,} ({losses / total * 100:.1f}%)")
    print(f"   Beraberlik:  {draws:>6,} ({draws / total * 100:.1f}%)")
    print(f"   Ortalama Ödül: {avg_reward:.4f}")


# ---- Ana Çalıştırma ----

if __name__ == "__main__":
    print("=" * 50)
    print("  BLACKJACK LAB - VERİ SETİ ÜRETİCİ")
    print("=" * 50)

    all_results = []

    # 1) Rastgele strateji
    results_random = simulate(100000, random_strategy, "random")
    print_stats(results_random, "Random")
    all_results.extend(results_random)

    # 2) Naive strateji (17'de dur)
    results_naive = simulate(100000, naive_strategy, "naive")
    print_stats(results_naive, "Naive")
    all_results.extend(results_naive)

    # 3) Basic Strategy
    results_basic = simulate(100000, basic_strategy, "basic")
    print_stats(results_basic, "Basic Strategy")
    all_results.extend(results_basic)

    # Hepsini tek CSV'ye kaydet
    save_to_csv(all_results, "dataset/games.csv")

    print("\n" + "=" * 50)
    print("  ÖZET KARŞILAŞTIRMA")
    print("=" * 50)

    for name, results in [("Random", results_random), ("Naive", results_naive), ("Basic", results_basic)]:
        total = len(results)
        win_rate = sum(1 for r in results if r["result"] in ("win", "blackjack")) / total * 100
        avg_reward = sum(r["reward"] for r in results) / total
        print(f"   {name:<15} Kazanma: {win_rate:>5.1f}%  |  Ort. Ödül: {avg_reward:>+.4f}")