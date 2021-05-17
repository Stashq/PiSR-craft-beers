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
  .demo-cell {
    height: 60px;
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: var(--mdc-theme-secondary, #333);
    color: var(--mdc-theme-on-secondary, #fff);
  }
  .head-cell {
    height: 60px;
    display: flex;
    justify-content: flex-end;
    align-items: center;
    background-color: var(--mdc-theme-secondary, #333);
    color: var(--mdc-theme-on-secondary, #fff);
  }
  .content-cell{
      height: 90vh;
      background-color: var(--mdc-theme-secondary, #333);
      display: flex;
  }

</style>