# Third-Party Dependencies

## DetaNet

- **Source**: <https://github.com/techandscixie2005/DetaNet>
- **Commit**: 4f92e64 (latest as of 2026-06-07)
- **License**: Not specified in repository (check with authors)
- **Integration**: Vendor snapshot (copied source, not a git submodule)
- **Files inspected**:
  - `detanet_model/detanet.py` — main model class
  - `detanet_model/modules/` — interaction blocks, embedding, radial basis

## Modifications

DetaNet source is **unchanged**. All MTO integration is via the adapter in
`src/mto/detanet_adapter.py`.
