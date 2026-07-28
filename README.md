# SEM Particle Analyzer

SEM görüntülerindeki parçacıkların boyutunu ölçer ve her parçacığı
**undercooled (sıvı)** ya da **katı** olarak sınıflandırır; katı olanların
desenini de belirler (Janus, Stripe, Composite, Lamellar). Ölçek, görüntünün
kendi scale bar'ından okunur — elle kalibrasyon gerekmez.

İnternet gerektirmez: tüm modeller uygulamanın içinde gelir.

---

## 🚀 İndir ve Kullan (kod bilmene gerek yok)

### Mac (Apple Silicon — M1 / M2 / M3 / M4)

1. Sağ üstteki **Releases** bölümüne git (ya da doğrudan:
   [github.com/EfeUtku0/SEM-Particle-Analyzer/releases](https://github.com/EfeUtku0/SEM-Particle-Analyzer/releases)).
2. En son sürümdeki **`SEM-Particle-Analyzer-macOS-AppleSilicon.zip`** dosyasını
   indir (~1.5 GB, biraz sürebilir).
3. İnen zip'e çift tıkla — içinden **SEM Particle Analyzer** uygulaması çıkar.
   İstersen `Applications` (Uygulamalar) klasörüne sürükle.
4. **İlk açılışta önemli adım:** uygulamaya **çift tıklama**. Onun yerine
   **sağ tıkla → Aç → Aç** de. (Çift tıklarsan "açılamıyor" uyarısı verir; bu
   uygulamanın bozuk olduğu anlamına gelmez — Apple imzasız uygulamalara böyle
   davranır. Bunu sadece bir kez yapman yeterli, sonra normal açılır.)
5. **İlk analiz** modeller yüklenirken ~1 dakika sürer. Sonrası hızlıdır.

> Mac hâlâ "uygulama zarar görmüş" derse: Terminal'i açıp şunu yapıştır ve Enter'a bas:
> ```bash
> xattr -cr "/Applications/SEM Particle Analyzer.app"
> ```
> (Uygulamayı Applications'a koymadıysan, yolunu ona göre değiştir.)

> **Not:** Bu paket yalnızca **Apple Silicon** Mac'lerde çalışır (2020 sonrası
> M-serisi işlemciler). Intel işlemcili eski Mac'lerde çalışmaz.

### Windows

Windows için hazır indirilebilir paket henüz yok — bir Windows bilgisayarda
derlenmesi gerekiyor (aşağıdaki *Building* bölümü). Hazır olduğunda yine
**Releases** bölümüne eklenecek.

---

## Verilerin nerede tutuluyor

Sol paneldeki klasör ve fotoğraf düzenin, analizlerin ve (varsa) yeniden
eğittiğin model, uygulamanın **dışında** saklanır — yani uygulamayı güncellemek
ya da değiştirmek bunlara asla dokunmaz:

| İşletim sistemi | Klasör |
|---|---|
| macOS | `~/Library/Application Support/SEM Particle Analyzer` |
| Windows | `%APPDATA%\SEM Particle Analyzer` |
| Linux | `~/.local/share/SEM Particle Analyzer` |

**Fotoğraflarının kendisi asla taşınmaz, adı değişmez, silinmez.** Uygulama
sadece onların yerini (yolunu) tutar. Sol paneldeki klasörler sanaldır: bir
fotoğrafı klasörler arasında sürüklemek diskte hiçbir şeyi değiştirmez.

### Yanlışlıkla bir şey silersen

Silme her zaman kasıtlıdır (sağ tık → Remove veya Delete tuşu; klasör silerken
onay sorulur). Yine de üç kademeli otomatik yedek tutulur:

- `library.json.bak` — en son değişiklikten önceki hâl
- `library.startup.json` — uygulamayı o gün açtığındaki hâl
- `library.shrink.json` — fotoğraf sayısının *azaldığı* andan önceki hâl

Geri yüklemek için: uygulamayı kapat, istediğin yedeği `library.json` olarak
yeniden adlandır, uygulamayı tekrar aç.

---

## Training modu (modeli düzeltmek)

🎓 **Training** butonu, parçacıklara tıklayıp doğru sınıfı vermeni ve işaretlediklerinle
desen modelini yeniden eğitmeni sağlar.

Bu tamamen **kendi bilgisayarına özeldir**: etiketlerin senin Masaüstü'ndeki
(Mac) / Belgeler'indeki `SEM Eğitim` klasörüne, eğittiğin model de kendi veri
klasörüne yazılır. Başkasının kopyasını etkilemez.

⚠️ **Yeniden eğitilen model, uygulamayla gelen modelin yerine geçer.** Uygulamayla
gelen model 32 fotoğraftaki binlerce etiketli parçacıkla eğitildi; sen sadece
birkaç tanesiyle yeniden eğitirsen sonuçlar *kötüleşir* ve nedeni belli olmaz.
Yalnızca ciddi miktarda etiketleme yaptıysan yeniden eğit. Uygulamayla gelen
modele dönmek için, yukarıdaki veri klasöründeki `patternnet.pt` dosyasını sil.

---

## Running from source (developers)

Python **3.11 or 3.12** recommended.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
.venv/bin/python app/gui.py                 # or double-click run.command / run.bat
```

## Building a distributable app

Build on the platform you're targeting (PyInstaller does not cross-compile):

```bash
.venv/bin/pyinstaller --noconfirm --clean sem_analyzer.spec
```

- macOS → `dist/SEM Particle Analyzer.app` — zip it with:
  `ditto -c -k --keepParent "dist/SEM Particle Analyzer.app" app-macos.zip`
- Windows → `dist/SEM Particle Analyzer/` — zip that whole folder to share it

The build embeds the Cellpose and EasyOCR weights from your `~/.cellpose` and
`~/.EasyOCR` folders. If the spec prints `WARNING: ... weights not found`, run
one analysis from source first so those models download, then rebuild —
otherwise the packaged app needs internet on its first run.

Verify a build without a GUI:

```bash
SEMPA_SELFTEST=/path/to/image.jpeg "dist/SEM Particle Analyzer.app/Contents/MacOS/SEM Particle Analyzer"
```
