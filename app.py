"""
Minimal Streamlit app for image steganography.

- Uses SteganoGAN CLI if installed (`steganogan encode` / `steganogan decode`).
- Falls back to a compact LSB embed/extract implementation (pure Python + Pillow + NumPy).
- Supports hiding text, images, video, or audio inside a cover image, and extracting them.
- Keeps UI tiny and synchronous.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
import streamlit as st
from PIL import Image
import numpy as np
import io


import subprocess
import shutil
import os
import tempfile
import json

# Fix PyTorch 2.6+ compatibility issue with steganogan
try:
    import torch
    # Monkey-patch torch.load to use weights_only=False by default
    # This is needed for steganogan compatibility with PyTorch 2.6+
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        # Set weights_only=False if not explicitly specified
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    # Also try adding safe globals as backup
    if hasattr(torch.serialization, 'add_safe_globals'):
        try:
            from steganogan.models import SteganoGAN
            torch.serialization.add_safe_globals([SteganoGAN])
        except ImportError:
            pass
except ImportError:
    pass  # torch not installed

st.set_page_config(page_title="Steganography", layout="centered")

st.title("Steganography")

# -----------------------
# Helper: LSB functions
# -----------------------
def _int_to_32bits(n: int):
    return [(n >> (8 * (3 - i))) & 0xFF for i in range(4)]

def _bytes_to_bitlist(b: bytes):
    return [ (byte >> i) & 1 for byte in b for i in reversed(range(8)) ]

def _bitlist_to_bytes(bits):
    assert len(bits) % 8 == 0
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i+8]:
            byte = (byte << 1) | bit
        out.append(byte)
    return bytes(out)

def lsb_embed_image_with_bytes(pil_img: Image.Image, data_bytes: bytes) -> Image.Image:
    """Embed data_bytes into an RGB image using simple LSB on each channel byte.
       Returns a new PIL.Image (RGB).
    """
    img = pil_img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    flat = arr.flatten()

    # prepend 32-bit length header (number of data bytes)
    length = len(data_bytes)
    header = bytes(_int_to_32bits(length))
    bits = _bytes_to_bitlist(header + data_bytes)
    needed = len(bits)
    if needed > flat.size:
        raise ValueError(f"Image too small: need {needed} bytes, available {flat.size} bytes.")

    # Clear LSB and set
    flat[:needed] = (flat[:needed] & 0xFE) | np.array(bits, dtype=np.uint8)
    new_arr = flat.reshape(arr.shape)
    return Image.fromarray(new_arr)

def lsb_extract_bytes_from_image(pil_img: Image.Image) -> bytes:
    """Extract a byte stream previously embedded with lsb_embed_image_with_bytes."""
    img = pil_img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    flat = arr.flatten()

    # Read first 32 bits -> 4 bytes -> length
    header_bits = [int(x & 1) for x in flat[:32]]
    header_bytes = _bitlist_to_bytes(header_bits)
    length = int.from_bytes(header_bytes, byteorder='big')
    total_bits = (4 + length) * 8
    if total_bits > flat.size:
        raise ValueError("Invalid or corrupted stego image (length larger than capacity).")

    data_bits = [int(x & 1) for x in flat[32: total_bits]]
    data = _bitlist_to_bytes(data_bits)
    return data

# -----------------------
# SteganoGAN helper (uses CLI and library)
# -----------------------
def has_steganogan_lib():
    """Check if steganogan can be imported as a library."""
    try:
        import steganogan
        return True
    except ImportError:
        return False

def has_steganogan_cli():
    return shutil.which("steganogan") is not None

def stegano_encode_lib(cover_img: Image.Image, message: str) -> tuple[bool, Image.Image, str]:
    """
    Use steganogan as a Python library to encode a message.
    Returns (success, stego_image, error_message).
    """
    try:
        from steganogan import SteganoGAN
        model = SteganoGAN.load()
        # Save cover image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cover_img.save(tmp.name, format='PNG')
            cover_path = tmp.name

        # Create output path
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        # Encode
        model.encode(cover_path, output_path, message)

        # Load result
        stego_img = Image.open(output_path)

        # Cleanup
        os.unlink(cover_path)
        os.unlink(output_path)

        return True, stego_img, "Success"
    except Exception as e:
        return False, None, str(e)

def stegano_decode_lib(stego_img: Image.Image) -> tuple[bool, str]:
    """
    Use steganogan as a Python library to decode a message.
    Returns (success, message).
    """
    try:
        from steganogan import SteganoGAN
        model = SteganoGAN.load()

        # Save stego image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            stego_img.save(tmp.name, format='PNG')
            stego_path = tmp.name

        # Decode
        message = model.decode(stego_path)

        # Cleanup
        os.unlink(stego_path)

        return True, message
    except Exception as e:
        return False, str(e)

def stegano_encode_cli(cover_path: str, message: str, output_path: str) -> tuple[bool, str]:
    """
    Call: steganogan encode <cover> "message"
    Some versions may support --output; to be robust we run the basic command,
    then check the created file (some implementations print file path).
    We'll call with --output if supported, else rely on default behavior and check output_path.
    Returns (success, stderr_or_stdout).
    """
    # Set environment to allow PyTorch to load weights without restriction
    env = os.environ.copy()
    env['TORCH_SERIALIZATION_SAFE_GLOBALS'] = '1'

    # Try with --output first (newer CLI)
    try_cmds = [
        ["steganogan", "encode", cover_path, message, "--output", output_path],
        ["steganogan", "encode", cover_path, message]
    ]
    for cmd in try_cmds:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
            out = (proc.stdout or "") + (proc.stderr or "")
            # If file created, success
            if os.path.exists(output_path):
                return True, out
            # sometimes CLI writes to stdout saying the output file name; treat that as success if file exists
        except Exception as e:
            out = str(e)
    # final fallback: check if a file with a common output name exists (e.g., cover_stego.png)
    # but we won't attempt guessing—return failure with last out text
    return False, out

def stegano_decode_cli(stego_path: str) -> tuple[bool, str]:
    """
    Call: steganogan decode <stego_image>
    CLI usually prints message to stdout.
    """
    # Set environment to allow PyTorch to load weights without restriction
    env = os.environ.copy()
    env['TORCH_SERIALIZATION_SAFE_GLOBALS'] = '1'

    try:
        proc = subprocess.run(["steganogan", "decode", stego_path], capture_output=True, text=True, timeout=30, env=env)
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0:
            # stdout usually contains the decoded message
            return True, proc.stdout.strip()
        else:
            return False, out
    except Exception as e:
        return False, str(e)

# -----------------------
# UI: embed
# -----------------------
st.header("Embed")
col1, col2 = st.columns(2)

with col1:
    cover_file = st.file_uploader("Upload cover image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    method = st.selectbox("Method", ["SteganoGAN"])

with col2:
    secret_type = st.selectbox("Secret type", ["text", "image", "video", "audio"])
    if secret_type == "text":
        secret_text = st.text_area("Secret text", height=120)
        secret_image_file = None
        secret_video_file = None
        secret_audio_file = None
    elif secret_type == "image":
        secret_text = None
        secret_image_file = st.file_uploader("Upload secret image", type=["png", "jpg", "jpeg"])
        secret_video_file = None
        secret_audio_file = None
    elif secret_type == "video":
        secret_text = None
        secret_image_file = None
        secret_video_file = st.file_uploader("Upload secret video", type=["mp4", "avi", "mov", "mkv", "flv", "wmv", "webm"])
        secret_audio_file = None
    else:  # audio
        secret_text = None
        secret_image_file = None
        secret_video_file = None
        secret_audio_file = st.file_uploader("Upload secret audio", type=["mp3", "wav", "flac", "m4a", "aac", "ogg", "wma"])

embed_btn = st.button("Embed")

if embed_btn:
    if not cover_file:
        st.error("Please upload a cover image first.")
    else:
        cover_bytes = cover_file.read()
        cover_img = Image.open(io.BytesIO(cover_bytes)).convert("RGB")

        # prepare data bytes
        data_bytes = None
        if secret_type == "text":
            if not secret_text:
                st.error("Please enter the secret text.")
            else:
                data_bytes = secret_text.encode("utf-8")
        else:
            if secret_type == "image":
                secret_file = secret_image_file
                error_msg = "Please upload a secret image."
            elif secret_type == "video":
                secret_file = secret_video_file
                error_msg = "Please upload a secret video."
            else:  # audio
                secret_file = secret_audio_file
                error_msg = "Please upload a secret audio."
            
            if not secret_file:
                st.error(error_msg)
            else:
                data_bytes = secret_file.read()

        if data_bytes and secret_type == "text" and method.startswith("SteganoGAN") and has_steganogan_lib():
            # use SteganoGAN library for text
            ok, stego_img, msg = stegano_encode_lib(cover_img, data_bytes.decode("utf-8", errors="ignore"))
            if ok and stego_img:
                buf = io.BytesIO()
                stego_img.save(buf, format="PNG")
                stego_bytes = buf.getvalue()
                st.success("Embedded with SteganoGAN (Library).")
                st.image(stego_img, caption="Stego image", use_column_width=True)
                st.download_button("Download stego image", stego_bytes, file_name="stego.png", mime="image/png")
            else:
                st.warning(f"SteganoGAN encode failed: {msg}. Falling back to LSB.")
                # fallthrough to LSB
                try:
                    stego_img = lsb_embed_image_with_bytes(cover_img, data_bytes)
                    buf = io.BytesIO()
                    stego_img.save(buf, format="PNG")
                    buf_bytes = buf.getvalue()
                    st.image(stego_img, caption="Stego image (LSB fallback)", use_column_width=True)
                    st.download_button("Download stego image (LSB)", buf_bytes, file_name="stego_lsb.png", mime="image/png")
                except Exception as e:
                    st.error("LSB embed failed: " + str(e))

        elif data_bytes and secret_type == "image":
            # Process image file (convert to PNG bytes)
            try:
                secret_img = Image.open(secret_image_file).convert("RGB")
                bio = io.BytesIO()
                secret_img.save(bio, format="PNG")
                data_bytes = bio.getvalue()
                
                stego_img = lsb_embed_image_with_bytes(cover_img, data_bytes)
                buf = io.BytesIO()
                stego_img.save(buf, format="PNG")
                buf_bytes = buf.getvalue()
                st.success("Embedded using LSB.")
                st.image(stego_img, caption="Stego image", use_column_width=True)
                st.download_button("Download stego image", buf_bytes, file_name="stego_lsb.png", mime="image/png")
            except Exception as e:
                st.error("LSB embed failed: " + str(e))

        elif data_bytes:
            # Use LSB embed (covers video, audio, or image-as-bytes)
            try:
                stego_img = lsb_embed_image_with_bytes(cover_img, data_bytes)
                buf = io.BytesIO()
                stego_img.save(buf, format="PNG")
                buf_bytes = buf.getvalue()
                st.success("Embedded using LSB.")
                st.image(stego_img, caption="Stego image", use_column_width=True)
                st.download_button("Download stego image", buf_bytes, file_name="stego_lsb.png", mime="image/png")
            except Exception as e:
                st.error("LSB embed failed: " + str(e))

# -----------------------
# UI: extract
# -----------------------
st.header("Extract")
stego_upload = st.file_uploader("Upload stego image to extract from", type=["png", "jpg", "jpeg"], key="extract")
extract_method = st.selectbox("Extract method", ["SteganoGAN"], key="extract_method")
extract_btn = st.button("Extract")

if extract_btn:
    if not stego_upload:
        st.error("Please upload a stego image first.")
    else:
        stego_bytes = stego_upload.read()
        stego_img = Image.open(io.BytesIO(stego_bytes)).convert("RGB")

        if extract_method.startswith("SteganoGAN") and has_steganogan_lib():
            # try library decode
            ok, out = stegano_decode_lib(stego_img)
            if ok:
                st.success("Decoded with SteganoGAN (Library).")
                st.text_area("Decoded message (SteganoGAN)", out, height=200)
            else:
                st.warning(f"SteganoGAN decode failed: {out}. Falling back to LSB.")
                # fallthrough to LSB below
                try:
                    data = lsb_extract_bytes_from_image(stego_img)
                    # try to decode as utf-8 text, else offer as file download
                    try:
                        text = data.decode("utf-8")
                        st.success("Decoded text (LSB).")
                        st.text_area("Decoded text", text, height=200)
                    except Exception:
                        st.success("Decoded raw bytes (LSB). Offering download.")
                        st.download_button("Download extracted bytes", data, file_name="extracted.bin")
                except Exception as e:
                    st.error("LSB extraction failed: " + str(e))
        else:
            # LSB extract
            try:
                data = lsb_extract_bytes_from_image(stego_img)
                # try utf-8
                try:
                    text = data.decode("utf-8")
                    st.success("Decoded text (LSB).")
                    st.text_area("Decoded text", text, height=200)
                except Exception:
                    # maybe it's an image stored as PNG bytes
                    # try to open with PIL
                    try:
                        img = Image.open(io.BytesIO(data))
                        st.success("Decoded an embedded image (LSB).")
                        st.image(img, caption="Extracted secret image", use_column_width=True)
                        out_buf = io.BytesIO()
                        img.save(out_buf, format="PNG")
                        st.download_button("Download extracted image", out_buf.getvalue(), file_name="extracted_secret.png", mime="image/png")
                    except Exception:
                        st.success("Decoded raw bytes (LSB). Offering download.")
                        st.download_button("Download extracted bytes", data, file_name="extracted.bin")
            except Exception as e:
                st.error("LSB extraction failed: " + str(e))
