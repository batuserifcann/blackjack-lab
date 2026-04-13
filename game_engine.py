import random


class Deck:
    """52 kartlık deste yönetimi."""

    SUITS = ["♠", "♥", "♦", "♣"]
    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

    def __init__(self, num_decks=1):
        self.num_decks = num_decks
        self.cards = []
        self.reset()

    def reset(self):
        """Desteyi yeniden oluştur ve karıştır."""
        self.cards = []
        for _ in range(self.num_decks):
            for suit in self.SUITS:
                for rank in self.RANKS:
                    self.cards.append((rank, suit))
        random.shuffle(self.cards)

    def deal(self):
        """Desteden bir kart çek."""
        if len(self.cards) == 0:
            self.reset()
        return self.cards.pop()
    

if __name__ == "__main__":
    deck = Deck()
    print(f"Destede {len(deck.cards)} kart var")
    print("İlk 5 kart:")
    for i in range(5):
        card = deck.deal()
        print(f"  {card[0]}{card[1]}")
    print(f"Kalan: {len(deck.cards)} kart")