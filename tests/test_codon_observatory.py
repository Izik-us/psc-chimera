import matplotlib

matplotlib.use("Agg")

from scripts.codon_observatory import CodonObservatory


def test_attention_capture_skips_long_sequences():
    observatory = CodonObservatory(update_every=3, max_attention_length=512)
    try:
        assert observatory.should_capture_attention(0, 512)
        assert observatory.should_capture_attention(3, 128)
        assert not observatory.should_capture_attention(1, 128)
        assert not observatory.should_capture_attention(3, 513)
    finally:
        observatory.close()


def test_visualizer_allocates_wide_landscape_panels():
    observatory = CodonObservatory()
    try:
        assert observatory.fig.get_figwidth() == 24
        assert (
            observatory.ax_decode.get_position().width
            > observatory.ax_loss.get_position().width
        )
        assert (
            observatory.ax_attention.get_position().width
            > observatory.ax_probs.get_position().width
        )
    finally:
        observatory.close()