# What to Upload to Google Drive

Create this folder structure in Google Drive, then share each folder and file as
**"Anyone with the link can view"**. Paste the resulting IDs into `site/drive_links.json`.

---

## Drive folder structure to create

```
📁 BorealisAI_UNICEF (root folder — share this)
├── 📁 maps
│   ├── nga_comparison_map.html          (238 MB)
│   ├── nga_predictions_map.html         (133 MB)
│   ├── nga_uncertainty_map.html         (134 MB)
│   ├── nga_predictions_map_sample.html  (  7 MB)
│   ├── nga_dimension_comparison_map.html( 16 MB)
│   ├── nga_dimension_shelter_map.html   (  5 MB)
│   ├── nga_dimension_sanitation_map.html(  5 MB)
│   ├── nga_dimension_water_map.html     (  5 MB)
│   ├── nga_dimension_nutrition_map.html (  5 MB)
│   ├── nga_dimension_edu_5_14_map.html  (  5 MB)
│   ├── nga_dimension_edu_15_17_map.html (  5 MB)
│   ├── nga_dimension_health_map.html    (  5 MB)
│   └── nga_dimension_health_36_59_map.html(5 MB)
├── 📁 docs
│   └── stakeholder_demo_script.html     (< 1 MB)
└── 📁 tables
    └── nga_lga_predictions.csv          (< 1 MB)
```

All files are in:
- `Data/outputs/nga/maps/`    ← all `nga_*.html` and `.geojson` files
- `docs/`                     ← `stakeholder_demo_script.html`
- `Data/outputs/nga/tables/`  ← CSVs

---

## How to get a file ID from a Drive share link

After right-clicking a file → Share → "Anyone with the link" → Copy link, the URL looks like:

```
https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                  This is the FILE_ID — copy just this part
```

For a **folder** link it looks like:
```
https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz?usp=sharing
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                        This is the FOLDER_ID
```

---

## After uploading

Paste all IDs into `site/drive_links.json` (the template is already there).
Then redeploy with:

```bash
npx vercel login     # only needed once
npx vercel --prod --yes
```

The portal will auto-build all links from that JSON file.
