<script>
    import LayoutGrid, { Cell, InnerGrid } from '@smui/layout-grid';
    import Menu from './User_menu.svelte'

    import BeerContent from './Beer_content.svelte';
    import BeerImages from './Beers.svelte'
  </script>
  

<LayoutGrid>
    <Cell span={12}>
        <div class="head-cell">
          <div class="title">
            <h1>CARPE CERVISIAM</h1>
          </div>
          <Menu></Menu>
        </div>
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
  .title h1{
    color:#e6a14e;
    margin: 40px;
    font-family: Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif;
  }
  :global(p){
    font-family: Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif;
  }
  :global(h2){
    font-family: Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif;
    letter-spacing: 2px;
    text-shadow: 2px 2px 5px rgb(0, 0, 0);
  }
  .head-cell {
    height: 60px;
    margin-top: -32px;
    width: 100%;
    top:0;
    display: flex;
    justify-content:space-between;
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
      box-shadow: #e6a24e72 0px 8px 16px -12px inset, #e6a24e72 0px 8px 16px -18px inset;
  }
  :global(body){
    background-color: #262c32;
  }
  

</style>