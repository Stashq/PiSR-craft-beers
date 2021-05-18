<script>
    import LayoutGrid, { Cell, InnerGrid } from '@smui/layout-grid';
    import Menu from './User_menu.svelte'

    import BeerContent from './Beer_content.svelte';
    import BeerImages from './Beers.svelte'
  </script>
  

<LayoutGrid>
    <Cell span={12}>
        <div class="head-cell"><Menu></Menu></div>
    </Cell>
    <Cell span={5}>
        <div class="content-cell">

            <BeerContent>
            </BeerContent>
        </div>
    </Cell>
    <Cell span={7}>
        <div class="content-cell">
            <BeerImages></BeerImages>
        </div>
    </Cell>
  </LayoutGrid>

<style>

  .head-cell {
    height: 60px;
    margin-top: -32px;
    width: 100%;
    top:0;
    display: flex;
    justify-content: flex-end;
    align-items: center;
    background-color: var(--mdc-theme-secondary, rgb(3, 3, 3));
    color: var(--mdc-theme-on-secondary, #fff);
    z-index: 10;
  }
  .content-cell{
      height: 90vh;
      margin-top: -32px;
      background-color: var(--mdc-theme-secondary, #333);
      display: flex;
      justify-content:center;
      z-index: 11;
  }

</style>