# Sample data

All files here were created by the book's author, Vasundra Srinivasan, with a Google generative video model. They show no real restaurant, kitchen, robot or person. Like the code, they are released under the MIT License (see [LICENSE](../../LICENSE)).

## Videos: `sample-videos/`

Seven clips of one Salmon Poke Bowl order, in pipeline order. Each is MP4 (H.264), 1280×720, 24 fps, 8.0 s.

| File | Shows | Size |
|---|---|---|
| `salmon-prep-1.mp4` | Starting the salmon preparation | 2.9 MB |
| `salmon-prep-2.mp4` | Cutting the salmon | 3.2 MB |
| `salmon-prep-3.mp4` | Portioning the salmon | 2.8 MB |
| `salmon-prep-4.mp4` | Finishing the salmon preparation | 2.5 MB |
| `salmon-bowl-assembly-5.mp4` | Assembling the bowl with rice and salmon | 2.6 MB |
| `quality-check-visual-6.mp4` | Visual quality check | 2.7 MB |
| `delivery-packaging-7.mp4` | Packaging for delivery | 2.1 MB |

The extractor reads the number after `salmon-prep-` as the prep stage, and the words `assembly`, `quality-check`, `packaging` and `delivery` as the video type, so keep the names if you add clips.

## Images

| File | Size | Used by |
|---|---|---|
| `reference_images/salmon_poke_bowl.jpg` | 2500×1667 | Quality checker (the reference for "Salmon Poke Bowl"); similarity demo (the query) |
| `sample_bowls/salmon_poke_bowl.jpg` | 2500×1667 | Similarity demo. The same file as the reference, so it scores 1.0000: a check that identical images match |
| `sample_bowls/miso_salmon_bowl.jpg` | 2500×1667 | Similarity demo |
| `sample_bowls/salmon_edamame_bowl.jpg` | 2500×1667 | Similarity demo |
| `sample_bowls/assembly_final.jpg` | 1280×720 | Similarity demo. A frame from `salmon-bowl-assembly-5.mp4` |
| `sample_bowls/final_packaging.jpg` | 1280×720 | Similarity demo. A frame from `delivery-packaging-7.mp4` |

The quality checker looks for `reference_images/<menu item in lower case, spaces as underscores>.jpg`, so a reference for "Miso Salmon Bowl" would be `reference_images/miso_salmon_bowl.jpg`.
