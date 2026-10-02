"""Inbound chat channels (Telegram today; iMessage / Slack later).

A channel adapter's whole job is to turn a provider's updates into `InboundMessage`s, post them to
the gateway's restricted `/channel/turn` endpoint, render the reply, and render confirmation cards.
Everything that decides what a message may DO — trust, taint, the restricted session, the
Guardian tiers, the audit trail — lives behind that endpoint, once, not per provider.
See docs/CHANNELS.md.
"""
