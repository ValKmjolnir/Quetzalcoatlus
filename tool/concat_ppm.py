from pathlib import Path

ppms = list(Path("ppms").glob("*/output.0.ppm"))


def read_ppm(path):
    with path.open("rb") as f:
        assert f.readline().strip() == b"P6"
        w, h = map(int, f.readline().split())
        assert f.readline().strip() == b"255"
        return w, h, f.read()


ppms.sort(key=lambda p: int(p.parent.name))
imgs = [read_ppm(p) for p in ppms]
W = sum(w for w, _, _ in imgs)
H = imgs[0][1]

with open("concat.ppm", "wb") as out:
    out.write(f"P6\n{W} {H}\n255\n".encode())
    for y in range(H):
        for w, _, data in imgs:
            out.write(data[y * w * 3:(y + 1) * w * 3])
