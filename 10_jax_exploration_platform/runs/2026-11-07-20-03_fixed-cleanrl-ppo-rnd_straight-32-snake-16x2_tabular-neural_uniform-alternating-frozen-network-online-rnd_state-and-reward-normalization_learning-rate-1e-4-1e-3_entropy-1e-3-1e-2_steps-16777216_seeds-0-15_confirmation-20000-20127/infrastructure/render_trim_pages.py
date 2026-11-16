import sys, fitz
doc = fitz.open('/p/rlprojects/RND/10_jax_exploration_platform/development_document/RND_JAX_exploration_development_document.pdf')
out = sys.argv[1]
hits = [i for i, page in enumerate(doc) if 'trimmed mean' in page.get_text()]
print('pages', [i+1 for i in hits], 'of', len(doc))
for i in hits:
    doc[i].get_pixmap(dpi=110).save(f'{out}/trim_page_{i+1}.png')
