def _render_single_card(self, card):
    # ... existing rendering code
    if card.type == 'small-pool':
        # Skip rendering the reason caption for small-pool cards
        pass
    else:
        # Render the reason caption for macro cards
        self.render_reason_caption(card.reason)
    # ... rest of the rendering code