-- List bands with Glam rock as main style and their lifespan until 2020
SELECT band_name,
       -- Calculate lifespan in years until split or 2020 if still active
       CASE
           WHEN split IS NULL OR split > 2020 THEN 2020 - formed
           ELSE split - formed
       END AS lifespan
FROM metal_bands
WHERE main_style = 'Glam rock'
ORDER BY lifespan DESC;
