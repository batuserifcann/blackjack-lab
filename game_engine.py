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
        self.cards = []
        for _ in range(self.num_decks):
            for suit in self.SUITS:
                for rank in self.RANKS:
                    self.cards.append((rank, suit))
        random.shuffle(self.cards)

    def deal(self):
        if len(self.cards) == 0:
            self.reset()
        return self.cards.pop()


def card_value(card):
    rank = card[0]
    if rank in ("J", "Q", "K"):
        return 10
    elif rank == "A":
        return 11
    else:
        return int(rank)


def hand_value(hand):
    total = 0
    aces = 0
    for card in hand:
        total += card_value(card)
        if card[0] == "A":
            aces += 1
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    usable_ace = aces > 0
    return total, usable_ace


class BlackjackGame:
    """Tek bir Blackjack eli yönetir."""

    def __init__(self, deck=None):
        self.deck = deck if deck else Deck()
        self.player_hand = []
        self.dealer_hand = []
        self.done = False
        self.result = None

    def deal_initial(self):
        self.player_hand = [self.deck.deal(), self.deck.deal()]
        self.dealer_hand = [self.deck.deal(), self.deck.deal()]
        self.done = False
        self.result = None

        player_total, _ = hand_value(self.player_hand)
        dealer_total, _ = hand_value(self.dealer_hand)

        if player_total == 21 and dealer_total == 21:
            self.done = True
            self.result = "draw"
        elif player_total == 21:
            self.done = True
            self.result = "blackjack"
        elif dealer_total == 21:
            self.done = True
            self.result = "lose"

    def player_hit(self):
        if self.done:
            return None, False
        card = self.deck.deal()
        self.player_hand.append(card)
        player_total, _ = hand_value(self.player_hand)
        if player_total > 21:
            self.done = True
            self.result = "lose"
            return card, True
        return card, False

    def player_stand(self):
        if self.done:
            return self.result
        dealer_total, _ = hand_value(self.dealer_hand)
        while dealer_total < 17:
            self.dealer_hand.append(self.deck.deal())
            dealer_total, _ = hand_value(self.dealer_hand)
        player_total, _ = hand_value(self.player_hand)
        if dealer_total > 21:
            self.result = "win"
        elif player_total > dealer_total:
            self.result = "win"
        elif player_total < dealer_total:
            self.result = "lose"
        else:
            self.result = "draw"
        self.done = True
        return self.result

    def play_one_hand(self, strategy_fn):
        self.deal_initial()
        actions_taken = []
        while not self.done:
            state = self.get_state()
            action = strategy_fn(state)
            if action == "hit":
                actions_taken.append("hit")
                card, bust = self.player_hit()
                if bust:
                    break
            else:
                actions_taken.append("stand")
                self.player_stand()
                break
        player_total, usable_ace = hand_value(self.player_hand)
        dealer_total, _ = hand_value(self.dealer_hand)
        return {
            "player_total": player_total,
            "dealer_total": dealer_total,
            "dealer_showing": card_value(self.dealer_hand[0]),
            "usable_ace": usable_ace,
            "num_hits": actions_taken.count("hit"),
            "actions": ",".join(actions_taken),
            "result": self.result,
            "reward": 1.5 if self.result == "blackjack" else (1 if self.result == "win" else (-1 if self.result == "lose" else 0)),
        }

    def get_state(self):
        player_total, usable_ace = hand_value(self.player_hand)
        dealer_showing = card_value(self.dealer_hand[0])
        return {
            "player_total": player_total,
            "dealer_showing": dealer_showing,
            "usable_ace": usable_ace,
            "done": self.done,
        }